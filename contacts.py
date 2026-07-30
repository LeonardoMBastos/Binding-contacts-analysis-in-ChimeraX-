# =============================================================================
# DockAnalyzer
# contacts.py
#
# Molecular contact detection and interpretation.
#
# This module identifies and organizes intermolecular contacts between
# receptor and ligand atoms. It provides high-level contact detection,
# residue grouping, interaction summaries and integration with DockModel.
#
# Unlike geometry.py, this module is chemically aware. It uses the
# geometric primitives implemented in geometry.py to classify molecular
# contacts and prepare them for subsequent interaction analyses
# (hydrogen bonds, hydrophobic interactions, pi interactions,
# salt bridges, scoring and reporting).
#
# Author
# ------
# Leonardo Marensi Bastos
#
# License
# -------
# MIT License
# =============================================================================

from __future__ import annotations

# =============================================================================
# Standard library imports
# =============================================================================

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import (
    Any, Callable, DefaultDict, Dict, Iterable, Iterator, List, Mapping,
    MutableMapping, Optional, Sequence, Set, Tuple, Union,
)

# =============================================================================
# Third-party imports
# =============================================================================

import numpy as np

# =============================================================================
# DockAnalyzer imports
# =============================================================================

if __package__:
    from . import config
    from . import geometry
    from . import utils
    from .geometry import (
        ContactGeometry,
    )

    from .utils import (
        DockLogger,
        DockModel,
    )

else:
    import config
    import geometry
    import utils
    from geometry import (
        ContactGeometry,
    )

    from utils import (
        DockLogger,
        DockModel,
    )


# =============================================================================
# Module metadata
# =============================================================================

__author__ = "Leonardo Marensi Bastos"
__license__ = "MIT"
__version__ = "1.0.0"

# =============================================================================
# Public module interface
# =============================================================================

__all__: List[str] = []


def _register_public_names(names: Iterable[str]) -> None:
    """Append unique public names while preserving declaration order."""

    known = set(__all__)
    for name in names:
        if name not in known:
            __all__.append(name)
            known.add(name)

# =============================================================================
# Type aliases
# =============================================================================

# -------------------------------------------------------------------------
# Numeric types
# -------------------------------------------------------------------------

Number = Union[int, float, np.integer, np.floating]
Coordinate = np.ndarray
CoordinateCollection = np.ndarray
Vector3D = np.ndarray
FloatArray = np.ndarray

# ChimeraX-compatible generic types.
AtomLike = Any
ResidueLike = Any
StructureLike = Any
ModelLike = Any
LigandLike = Any
ReceptorLike = Any

AtomCollection = Sequence[AtomLike]
ResidueCollection = Sequence[ResidueLike]
ContactCollection = Sequence[ContactGeometry]
Metadata = Dict[str, Any]
Statistics = Dict[str, Any]
ContactDictionary = Dict[str, ContactGeometry]
ResidueContactDictionary = Dict[ResidueLike, List[ContactGeometry]]

# -------------------------------------------------------------------------
# Section 1 public interface
# -------------------------------------------------------------------------

_SECTION_1_PUBLIC_NAMES = [
    "Number",
    "Coordinate",
    "CoordinateCollection",
    "Vector3D",
    "FloatArray",
    "AtomLike",
    "ResidueLike",
    "StructureLike",
    "ModelLike",
    "LigandLike",
    "ReceptorLike",
    "AtomCollection",
    "ResidueCollection",
    "ContactCollection",
    "Metadata",
    "Statistics",
    "ContactDictionary",
    "ResidueContactDictionary",
]

_register_public_names(_SECTION_1_PUBLIC_NAMES)

# =============================================================================
# Internal constants
# =============================================================================

_MODULE_NAME = "contacts"

_LOGGER = DockLogger(_MODULE_NAME)

# =============================================================================
# End of Section 1
# ============================================================================= 

# =============================================================================
# Section 2 — Internal constants and contact-specific types
# =============================================================================

# -----------------------------------------------------------------------------
# Contact classification labels
# -----------------------------------------------------------------------------

CONTACT_TYPE_UNKNOWN = "unknown"

CONTACT_TYPE_CONTACT = "contact"

CONTACT_TYPE_CLOSE_CONTACT = "close_contact"

CONTACT_TYPE_VDW = "van_der_waals"

CONTACT_TYPE_CLASH = "steric_clash"

CONTACT_TYPE_SELF = "self"

_CONTACT_TYPES = frozenset(
    {
        CONTACT_TYPE_UNKNOWN,
        CONTACT_TYPE_CONTACT,
        CONTACT_TYPE_CLOSE_CONTACT,
        CONTACT_TYPE_VDW,
        CONTACT_TYPE_CLASH,
        CONTACT_TYPE_SELF,
    }
)

# -----------------------------------------------------------------------------
# Internal defaults
# -----------------------------------------------------------------------------

DEFAULT_METADATA_KEY = "contacts"

DEFAULT_RESIDUE_SEPARATOR = ":"

DEFAULT_CONTACT_ID_SEPARATOR = "|"

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

ContactPair = Tuple[
    AtomLike,
    AtomLike,
]

IndexedContactPair = Tuple[
    int,
    int,
]

AtomIndex = int

ResidueContactKey = Tuple[
    str,
    int,
    str,
]

ContactIdentifier = str

ContactClassification = str

# -----------------------------------------------------------------------------
# Internal immutable empty objects
# -----------------------------------------------------------------------------

_EMPTY_METADATA: Mapping[str, Any] = {}

_EMPTY_CONTACT_LIST: Tuple[ContactGeometry, ...] = ()

_EMPTY_ATOM_LIST: Tuple[AtomLike, ...] = ()

# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

_VALID_CONTACT_CLASSIFICATIONS = frozenset(
    _CONTACT_TYPES
)

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_2_PUBLIC_NAMES = [
    "CONTACT_TYPE_UNKNOWN",
    "CONTACT_TYPE_CONTACT",
    "CONTACT_TYPE_CLOSE_CONTACT",
    "CONTACT_TYPE_VDW",
    "CONTACT_TYPE_CLASH",
    "CONTACT_TYPE_SELF",
    "ContactPair",
    "IndexedContactPair",
    "AtomIndex",
    "ResidueContactKey",
    "ContactIdentifier",
    "ContactClassification",
]

_register_public_names(_SECTION_2_PUBLIC_NAMES)

# =============================================================================
# End of Section 2
# =============================================================================


# =============================================================================
# Section 3 — Result structures
# =============================================================================


# -----------------------------------------------------------------------------
# Atom-level contact result
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomContact:
    """
    Represent a molecular contact between two atoms.

    Parameters
    ----------
    atom_1 : AtomLike
        First atom, normally belonging to the ligand.
    atom_2 : AtomLike
        Second atom, normally belonging to the receptor.
    geometry : ContactGeometry
        Geometric description of the contact.
    classification : ContactClassification, optional
        General contact classification.
    atom_1_index : int or None, optional
        Index of the first atom in its original collection.
    atom_2_index : int or None, optional
        Index of the second atom in its original collection.
    residue_1 : ResidueLike or None, optional
        Residue associated with the first atom.
    residue_2 : ResidueLike or None, optional
        Residue associated with the second atom.
    metadata : Mapping[str, Any], optional
        Additional information associated with the contact.

    Notes
    -----
    This structure represents a chemically interpreted contact.

    The underlying geometric information remains stored in
    :class:`geometry.ContactGeometry`.
    """

    atom_1: AtomLike

    atom_2: AtomLike

    geometry: ContactGeometry

    classification: ContactClassification = (
        CONTACT_TYPE_CONTACT
    )

    atom_1_index: Optional[int] = None

    atom_2_index: Optional[int] = None

    residue_1: Optional[ResidueLike] = None

    residue_2: Optional[ResidueLike] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize the atom-contact result.

        Raises
        ------
        TypeError
            If the geometry is not a ContactGeometry instance.
        ValueError
            If indices or classification values are invalid.
        """

        if not isinstance(
            self.geometry,
            ContactGeometry,
        ):
            raise TypeError(
                "geometry must be a "
                "ContactGeometry instance."
            )

        classification = str(
            self.classification
        ).strip().lower()

        if not classification:
            raise ValueError(
                "classification cannot be empty."
            )

        if classification not in (
            _VALID_CONTACT_CLASSIFICATIONS
        ):
            classification = (
                CONTACT_TYPE_UNKNOWN
            )

        object.__setattr__(
            self,
            "classification",
            classification,
        )

        for attribute_name in (
            "atom_1_index",
            "atom_2_index",
        ):
            value = getattr(
                self,
                attribute_name,
            )

            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (
                    int,
                    np.integer,
                ),
            ):
                raise TypeError(
                    f"{attribute_name} must be "
                    "an integer or None."
                )

            integer_value = int(
                value
            )

            if integer_value < 0:
                raise ValueError(
                    f"{attribute_name} cannot "
                    "be negative."
                )

            object.__setattr__(
                self,
                attribute_name,
                integer_value,
            )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )

    @property
    def distance(
        self,
    ) -> np.float64:
        """
        Return the interatomic distance.

        Returns
        -------
        numpy.float64
            Contact distance.
        """

        return np.float64(
            self.geometry.distance
        )

    @property
    def cutoff(
        self,
    ) -> np.float64:
        """
        Return the contact cutoff.

        Returns
        -------
        numpy.float64
            Contact cutoff.
        """

        return np.float64(
            self.geometry.cutoff
        )

    @property
    def is_contact(
        self,
    ) -> bool:
        """
        Return whether the pair satisfies the contact cutoff.

        Returns
        -------
        bool
            ``True`` when the pair is a contact.
        """

        return bool(
            self.geometry.is_contact
        )

    @property
    def is_clash(
        self,
    ) -> bool:
        """
        Return whether the contact is classified as a steric clash.

        Returns
        -------
        bool
            ``True`` for steric clashes.
        """

        return (
            self.classification
            == CONTACT_TYPE_CLASH
        )

    @property
    def atom_pair(
        self,
    ) -> ContactPair:
        """
        Return the interacting atom pair.

        Returns
        -------
        tuple
            ``(atom_1, atom_2)``.
        """

        return (
            self.atom_1,
            self.atom_2,
        )

    @property
    def index_pair(
        self,
    ) -> Optional[
        IndexedContactPair
    ]:
        """
        Return the collection index pair when available.

        Returns
        -------
        tuple of int or None
            Atom indices or ``None`` when one index is missing.
        """

        if (
            self.atom_1_index is None
            or self.atom_2_index is None
        ):
            return None

        return (
            self.atom_1_index,
            self.atom_2_index,
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
        include_geometry: bool = True,
        include_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize the atom contact.

        Parameters
        ----------
        include_atoms : bool, optional
            Whether raw atom objects should be included.
        include_geometry : bool, optional
            Whether the geometric result should be included.
        include_coordinates : bool, optional
            Whether coordinates should be included in the geometry
            serialization.

        Returns
        -------
        dict
            Serializable representation of the contact.
        """

        result: Dict[str, Any] = {
            "classification": (
                self.classification
            ),
            "distance": float(
                self.distance
            ),
            "cutoff": float(
                self.cutoff
            ),
            "is_contact": (
                self.is_contact
            ),
            "is_clash": (
                self.is_clash
            ),
            "atom_1_index": (
                self.atom_1_index
            ),
            "atom_2_index": (
                self.atom_2_index
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_geometry:
            result[
                "geometry"
            ] = self.geometry.to_dict(
                include_coordinates=(
                    include_coordinates
                ),
                include_atoms=False,
            )

        if include_atoms:
            result[
                "atom_1"
            ] = self.atom_1

            result[
                "atom_2"
            ] = self.atom_2

            result[
                "residue_1"
            ] = self.residue_1

            result[
                "residue_2"
            ] = self.residue_2

        return result


# -----------------------------------------------------------------------------
# Residue-level contact result
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidueContact:
    """
    Represent all contacts associated with one receptor residue.

    Parameters
    ----------
    residue : ResidueLike
        Residue represented by this result.
    key : ResidueContactKey
        Stable residue identifier.
    contacts : sequence of AtomContact
        Atom-level contacts involving the residue.
    minimum_distance : float or None, optional
        Minimum contact distance. If omitted, it is calculated from
        ``contacts``.
    metadata : Mapping[str, Any], optional
        Additional residue-level information.
    """

    residue: ResidueLike

    key: ResidueContactKey

    contacts: Sequence[
        AtomContact
    ] = field(
        default_factory=tuple
    )

    minimum_distance: Optional[
        float
    ] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize the residue-contact result.

        Raises
        ------
        TypeError
            If the residue key or contact collection is invalid.
        ValueError
            If a distance is negative or non-finite.
        """

        if (
            not isinstance(
                self.key,
                tuple,
            )
            or len(
                self.key
            )
            != 3
        ):
            raise TypeError(
                "key must be a three-item "
                "ResidueContactKey tuple."
            )

        residue_name = str(
            self.key[
                0
            ]
        )

        residue_number = self.key[
            1
        ]

        chain_id = str(
            self.key[
                2
            ]
        )

        if isinstance(
            residue_number,
            bool,
        ) or not isinstance(
            residue_number,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "Residue number in key must "
                "be an integer."
            )

        normalized_key: ResidueContactKey = (
            residue_name,
            int(
                residue_number
            ),
            chain_id,
        )

        object.__setattr__(
            self,
            "key",
            normalized_key,
        )

        normalized_contacts = tuple(
            self.contacts
        )

        for contact in normalized_contacts:
            if not isinstance(
                contact,
                AtomContact,
            ):
                raise TypeError(
                    "contacts must contain only "
                    "AtomContact instances."
                )

        object.__setattr__(
            self,
            "contacts",
            normalized_contacts,
        )

        minimum_distance = (
            self.minimum_distance
        )

        if minimum_distance is None:
            if normalized_contacts:
                minimum_distance = min(
                    contact.distance
                    for contact in (
                        normalized_contacts
                    )
                )

        if minimum_distance is not None:
            minimum_distance_value = (
                np.float64(
                    minimum_distance
                )
            )

            if (
                not np.isfinite(
                    minimum_distance_value
                )
                or minimum_distance_value
                < 0.0
            ):
                raise ValueError(
                    "minimum_distance must be "
                    "finite and non-negative."
                )

            object.__setattr__(
                self,
                "minimum_distance",
                minimum_distance_value,
            )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )

    @property
    def contact_count(
        self,
    ) -> int:
        """
        Return the number of atom-level contacts.

        Returns
        -------
        int
            Contact count.
        """

        return len(
            self.contacts
        )

    @property
    def atom_contacts(
        self,
    ) -> Tuple[
        AtomContact,
        ...,
    ]:
        """
        Return contacts as an immutable tuple.

        Returns
        -------
        tuple of AtomContact
            Atom-level contacts.
        """

        return tuple(
            self.contacts
        )

    @property
    def classifications(
        self,
    ) -> Tuple[
        ContactClassification,
        ...,
    ]:
        """
        Return unique contact classifications.

        Returns
        -------
        tuple of str
            Sorted contact classifications.
        """

        return tuple(
            sorted(
                {
                    contact.classification
                    for contact in self.contacts
                }
            )
        )

    @property
    def has_clash(
        self,
    ) -> bool:
        """
        Return whether the residue has at least one steric clash.

        Returns
        -------
        bool
            Clash status.
        """

        return any(
            contact.is_clash
            for contact in self.contacts
        )

    @property
    def mean_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the mean contact distance.

        Returns
        -------
        numpy.float64 or None
            Mean distance, or ``None`` when no contacts are present.
        """

        if not self.contacts:
            return None

        return np.float64(
            np.mean(
                [
                    contact.distance
                    for contact in (
                        self.contacts
                    )
                ],
                dtype=np.float64,
            )
        )

    def to_dict(
        self,
        *,
        include_contacts: bool = True,
        include_atoms: bool = False,
        include_residue: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the residue-level result.

        Parameters
        ----------
        include_contacts : bool, optional
            Whether atom-level contacts should be included.
        include_atoms : bool, optional
            Whether raw atom objects should be included in contacts.
        include_residue : bool, optional
            Whether the raw residue object should be included.

        Returns
        -------
        dict
            Serializable residue-contact representation.
        """

        result: Dict[str, Any] = {
            "key": {
                "residue_name": (
                    self.key[
                        0
                    ]
                ),
                "residue_number": (
                    self.key[
                        1
                    ]
                ),
                "chain_id": (
                    self.key[
                        2
                    ]
                ),
            },
            "contact_count": (
                self.contact_count
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
            "classifications": list(
                self.classifications
            ),
            "has_clash": (
                self.has_clash
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_contacts:
            result[
                "contacts"
            ] = [
                contact.to_dict(
                    include_atoms=(
                        include_atoms
                    )
                )
                for contact in self.contacts
            ]

        if include_residue:
            result[
                "residue"
            ] = self.residue

        return result


# -----------------------------------------------------------------------------
# Complete contact-analysis result
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactAnalysisResult:
    """
    Store the complete result of a contact analysis.

    Parameters
    ----------
    contacts : sequence of AtomContact
        All detected atom-level contacts.
    residue_contacts : sequence of ResidueContact
        Contacts grouped by receptor residue.
    ligand_atoms : sequence of AtomLike, optional
        Ligand atoms included in the analysis.
    receptor_atoms : sequence of AtomLike, optional
        Receptor atoms included in the analysis.
    cutoff : float or None, optional
        Contact cutoff used during the analysis.
    statistics : Mapping[str, Any], optional
        Analysis statistics.
    metadata : Mapping[str, Any], optional
        Additional analysis metadata.
    """

    contacts: Sequence[
        AtomContact
    ] = field(
        default_factory=tuple
    )

    residue_contacts: Sequence[
        ResidueContact
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

    cutoff: Optional[
        float
    ] = None

    statistics: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize the complete analysis result.

        Raises
        ------
        TypeError
            If contact collections contain invalid result objects.
        ValueError
            If the cutoff is invalid.
        """

        contacts = tuple(
            self.contacts
        )

        residue_contacts = tuple(
            self.residue_contacts
        )

        for contact in contacts:
            if not isinstance(
                contact,
                AtomContact,
            ):
                raise TypeError(
                    "contacts must contain only "
                    "AtomContact instances."
                )

        for residue_contact in (
            residue_contacts
        ):
            if not isinstance(
                residue_contact,
                ResidueContact,
            ):
                raise TypeError(
                    "residue_contacts must contain "
                    "only ResidueContact instances."
                )

        object.__setattr__(
            self,
            "contacts",
            contacts,
        )

        object.__setattr__(
            self,
            "residue_contacts",
            residue_contacts,
        )

        object.__setattr__(
            self,
            "ligand_atoms",
            tuple(
                self.ligand_atoms
            ),
        )

        object.__setattr__(
            self,
            "receptor_atoms",
            tuple(
                self.receptor_atoms
            ),
        )

        if self.cutoff is not None:
            cutoff_value = np.float64(
                self.cutoff
            )

            if (
                not np.isfinite(
                    cutoff_value
                )
                or cutoff_value
                <= 0.0
            ):
                raise ValueError(
                    "cutoff must be finite "
                    "and greater than zero."
                )

            object.__setattr__(
                self,
                "cutoff",
                cutoff_value,
            )

        object.__setattr__(
            self,
            "statistics",
            dict(
                self.statistics
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )

    @property
    def contact_count(
        self,
    ) -> int:
        """
        Return the total number of atom contacts.

        Returns
        -------
        int
            Atom-contact count.
        """

        return len(
            self.contacts
        )

    @property
    def residue_count(
        self,
    ) -> int:
        """
        Return the number of contacting residues.

        Returns
        -------
        int
            Contacting-residue count.
        """

        return len(
            self.residue_contacts
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
        Return the shortest detected contact distance.

        Returns
        -------
        numpy.float64 or None
            Minimum distance, or ``None`` if no contacts exist.
        """

        if not self.contacts:
            return None

        return np.float64(
            min(
                contact.distance
                for contact in self.contacts
            )
        )

    @property
    def mean_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the mean contact distance.

        Returns
        -------
        numpy.float64 or None
            Mean contact distance, or ``None`` if no contacts exist.
        """

        if not self.contacts:
            return None

        return np.float64(
            np.mean(
                [
                    contact.distance
                    for contact in (
                        self.contacts
                    )
                ],
                dtype=np.float64,
            )
        )

    @property
    def clash_count(
        self,
    ) -> int:
        """
        Return the number of steric clashes.

        Returns
        -------
        int
            Steric-clash count.
        """

        return sum(
            1
            for contact in self.contacts
            if contact.is_clash
        )

    @property
    def has_contacts(
        self,
    ) -> bool:
        """
        Return whether at least one contact was detected.

        Returns
        -------
        bool
            Contact-detection status.
        """

        return bool(
            self.contacts
        )

    @property
    def has_clashes(
        self,
    ) -> bool:
        """
        Return whether at least one steric clash was detected.

        Returns
        -------
        bool
            Clash-detection status.
        """

        return self.clash_count > 0

    @property
    def contacting_residue_keys(
        self,
    ) -> Tuple[
        ResidueContactKey,
        ...,
    ]:
        """
        Return stable identifiers of contacting residues.

        Returns
        -------
        tuple of ResidueContactKey
            Residue keys.
        """

        return tuple(
            residue_contact.key
            for residue_contact in (
                self.residue_contacts
            )
        )

    def contacts_by_classification(
        self,
        classification: ContactClassification,
    ) -> Tuple[
        AtomContact,
        ...,
    ]:
        """
        Return contacts matching a classification.

        Parameters
        ----------
        classification : str
            Contact classification.

        Returns
        -------
        tuple of AtomContact
            Matching contacts.
        """

        normalized_classification = str(
            classification
        ).strip().lower()

        return tuple(
            contact
            for contact in self.contacts
            if contact.classification
            == normalized_classification
        )

    def get_residue_contact(
        self,
        key: ResidueContactKey,
    ) -> Optional[
        ResidueContact
    ]:
        """
        Retrieve a residue-contact result by key.

        Parameters
        ----------
        key : ResidueContactKey
            Residue identifier.

        Returns
        -------
        ResidueContact or None
            Matching result, or ``None`` when not found.
        """

        for residue_contact in (
            self.residue_contacts
        ):
            if residue_contact.key == key:
                return residue_contact

        return None

    def to_dict(
        self,
        *,
        include_contacts: bool = True,
        include_residue_contacts: bool = True,
        include_atoms: bool = False,
        include_atom_collections: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the complete contact-analysis result.

        Parameters
        ----------
        include_contacts : bool, optional
            Whether atom-level contacts should be included.
        include_residue_contacts : bool, optional
            Whether residue-grouped contacts should be included.
        include_atoms : bool, optional
            Whether raw atom and residue objects should be included.
        include_atom_collections : bool, optional
            Whether the complete ligand and receptor atom collections
            should be included.

        Returns
        -------
        dict
            Serializable analysis result.
        """

        result: Dict[str, Any] = {
            "contact_count": (
                self.contact_count
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
            "clash_count": (
                self.clash_count
            ),
            "has_contacts": (
                self.has_contacts
            ),
            "has_clashes": (
                self.has_clashes
            ),
            "cutoff": (
                None
                if self.cutoff is None
                else float(
                    self.cutoff
                )
            ),
            "statistics": dict(
                self.statistics
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_contacts:
            result[
                "contacts"
            ] = [
                contact.to_dict(
                    include_atoms=(
                        include_atoms
                    )
                )
                for contact in self.contacts
            ]

        if include_residue_contacts:
            result[
                "residue_contacts"
            ] = [
                residue_contact.to_dict(
                    include_contacts=(
                        include_contacts
                    ),
                    include_atoms=(
                        include_atoms
                    ),
                    include_residue=(
                        include_atoms
                    ),
                )
                for residue_contact in (
                    self.residue_contacts
                )
            ]

        if include_atom_collections:
            result[
                "ligand_atoms"
            ] = list(
                self.ligand_atoms
            )

            result[
                "receptor_atoms"
            ] = list(
                self.receptor_atoms
            )

        return result


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_3_PUBLIC_NAMES = [
    "AtomContact",
    "ResidueContact",
    "ContactAnalysisResult",
]

_register_public_names(_SECTION_3_PUBLIC_NAMES)


# =============================================================================
# End of Section 3
# =============================================================================


# =============================================================================
# Section 4 — Atom identification and validation
# =============================================================================


# -----------------------------------------------------------------------------
# Generic attribute extraction
# -----------------------------------------------------------------------------

def _get_object_value(
    object_: Any,
    names: Sequence[str],
    *,
    default: Any = None,
    call_if_callable: bool = False,
) -> Any:
    """Retrieve the first available value from an object or mapping."""

    if object_ is None:
        return default

    for name in names:
        value = default
        found = False

        if isinstance(
            object_,
            Mapping,
        ):
            if name in object_:
                value = object_[
                    name
                ]

                found = True

        else:
            try:
                value = getattr(
                    object_,
                    name,
                )

                found = True

            except (
                AttributeError,
                TypeError,
            ):
                found = False

        if not found:
            continue

        if (
            call_if_callable
            and callable(
                value
            )
        ):
            try:
                value = value()

            except TypeError:
                continue

        if value is not None:
            return value

    return default


def _normalize_text_value(
    value: Any,
    *,
    default: str = "",
    uppercase: bool = False,
) -> str:
    """Normalize a value as stripped text."""

    if value is None:
        text = str(
            default
        )

    else:
        try:
            text = str(
                value
            )

        except Exception:
            text = str(
                default
            )

    text = text.strip()

    if not text:
        text = str(
            default
        ).strip()

    if uppercase:
        text = text.upper()

    return text


# -----------------------------------------------------------------------------
# Atom name
# -----------------------------------------------------------------------------

def get_atom_name(
    atom: AtomLike,
    *,
    default: str = "",
    uppercase: bool = False,
) -> str:
    """
    Retrieve an atom name safely.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : str, optional
        Value returned when no atom name is available.
    uppercase : bool, optional
        Whether the returned name should be uppercase.

    Returns
    -------
    str
        Atom name.

    Examples
    --------
    Synthetic mapping:

    ``get_atom_name({"name": "CA"})``

    Synthetic object:

    ``get_atom_name(atom)`` where ``atom.name`` is defined.
    """

    value = _get_object_value(
        atom,
        (
            "name",
            "atom_name",
            "label",
        ),
        default=default,
    )

    return _normalize_text_value(
        value,
        default=default,
        uppercase=uppercase,
    )


# -----------------------------------------------------------------------------
# Element identification
# -----------------------------------------------------------------------------

def _extract_element_symbol(
    element: Any,
    *,
    default: str = "",
) -> str:
    """Extract a chemical element symbol from an element-like value."""

    if element is None:
        return _normalize_text_value(
            default,
            uppercase=True,
        )

    if isinstance(
        element,
        str,
    ):
        return _normalize_text_value(
            element,
            default=default,
            uppercase=True,
        )

    if isinstance(
        element,
        (
            int,
            np.integer,
        ),
    ):
        atomic_number = int(
            element
        )

        atomic_number_symbols = {
            1: "H",
            5: "B",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            11: "NA",
            12: "MG",
            15: "P",
            16: "S",
            17: "CL",
            19: "K",
            20: "CA",
            25: "MN",
            26: "FE",
            27: "CO",
            28: "NI",
            29: "CU",
            30: "ZN",
            35: "BR",
            53: "I",
        }

        return atomic_number_symbols.get(
            atomic_number,
            _normalize_text_value(
                default,
                uppercase=True,
            ),
        )

    symbol = _get_object_value(
        element,
        (
            "symbol",
            "name",
            "element",
            "element_name",
        ),
        default=None,
    )

    if symbol is not None:
        return _normalize_text_value(
            symbol,
            default=default,
            uppercase=True,
        )

    atomic_number = _get_object_value(
        element,
        (
            "number",
            "atomic_number",
        ),
        default=None,
    )

    if atomic_number is not None:
        try:
            return _extract_element_symbol(
                int(
                    atomic_number
                ),
                default=default,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            pass

    return _normalize_text_value(
        default,
        uppercase=True,
    )


def infer_element_from_atom_name(
    atom_name: Any,
    *,
    default: str = "",
) -> str:
    """
    Infer an element symbol from an atom name.

    Parameters
    ----------
    atom_name : Any
        Atom name.
    default : str, optional
        Fallback element symbol.

    Returns
    -------
    str
        Inferred uppercase element symbol.

    Notes
    -----
    This is a fallback heuristic. Explicit element information should always
    be preferred when available.
    """

    name = _normalize_text_value(
        atom_name,
        uppercase=True,
    )

    if not name:
        return _normalize_text_value(
            default,
            uppercase=True,
        )

    stripped_name = name.lstrip(
        "0123456789"
    )

    if not stripped_name:
        return _normalize_text_value(
            default,
            uppercase=True,
        )

    two_letter_elements = {
        "BR",
        "CL",
        "FE",
        "MG",
        "MN",
        "NA",
        "NI",
        "ZN",
        "CA",
        "CO",
        "CU",
    }

    first_two = stripped_name[
        :2
    ]

    if first_two in two_letter_elements:
        return first_two

    return stripped_name[
        0
    ]


def get_atom_element(
    atom: AtomLike,
    *,
    default: str = "",
    infer_from_name: bool = True,
) -> str:
    """
    Retrieve the chemical element of an atom safely.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : str, optional
        Value returned when the element is unavailable.
    infer_from_name : bool, optional
        Whether the element may be inferred from the atom name.

    Returns
    -------
    str
        Uppercase chemical element symbol.
    """

    element = _get_object_value(
        atom,
        (
            "element",
            "element_symbol",
            "symbol",
            "atomic_number",
        ),
        default=None,
    )

    symbol = _extract_element_symbol(
        element,
        default="",
    )

    if symbol:
        return symbol

    if infer_from_name:
        return infer_element_from_atom_name(
            get_atom_name(
                atom
            ),
            default=default,
        )

    return _normalize_text_value(
        default,
        uppercase=True,
    )


def get_atom_atomic_number(
    atom: AtomLike,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Retrieve an atom's atomic number.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : int or None, optional
        Value returned when the atomic number cannot be determined.

    Returns
    -------
    int or None
        Atomic number.
    """

    element = _get_object_value(
        atom,
        (
            "element",
        ),
        default=None,
    )

    atomic_number = _get_object_value(
        element,
        (
            "number",
            "atomic_number",
        ),
        default=None,
    )

    if atomic_number is None:
        atomic_number = _get_object_value(
            atom,
            (
                "atomic_number",
                "element_number",
            ),
            default=None,
        )

    if atomic_number is not None:
        try:
            value = int(
                atomic_number
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            value = None

        if (
            value is not None
            and value > 0
        ):
            return value

    symbol_to_number = {
        "H": 1,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "NA": 11,
        "MG": 12,
        "P": 15,
        "S": 16,
        "CL": 17,
        "K": 19,
        "CA": 20,
        "MN": 25,
        "FE": 26,
        "CO": 27,
        "NI": 28,
        "CU": 29,
        "ZN": 30,
        "BR": 35,
        "I": 53,
    }

    return symbol_to_number.get(
        get_atom_element(
            atom,
            default="",
        ),
        default,
    )


# -----------------------------------------------------------------------------
# Residue and structure retrieval
# -----------------------------------------------------------------------------

def get_atom_residue(
    atom: AtomLike,
    *,
    default: Optional[ResidueLike] = None,
) -> Optional[ResidueLike]:
    """
    Retrieve the residue associated with an atom.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : ResidueLike or None, optional
        Value returned when no residue is associated with the atom.

    Returns
    -------
    ResidueLike or None
        Associated residue.
    """

    return _get_object_value(
        atom,
        (
            "residue",
            "parent_residue",
        ),
        default=default,
    )


def get_atom_structure(
    atom: AtomLike,
    *,
    default: Optional[StructureLike] = None,
) -> Optional[StructureLike]:
    """
    Retrieve the molecular structure associated with an atom.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : StructureLike or None, optional
        Value returned when no structure can be determined.

    Returns
    -------
    StructureLike or None
        Associated molecular structure.

    Notes
    -----
    The structure is first retrieved directly from the atom. If unavailable,
    the residue associated with the atom is inspected.
    """

    structure = _get_object_value(
        atom,
        (
            "structure",
            "model",
            "molecule",
            "parent_structure",
        ),
        default=None,
    )

    if structure is not None:
        return structure

    residue = get_atom_residue(
        atom
    )

    if residue is None:
        return default

    return _get_object_value(
        residue,
        (
            "structure",
            "model",
            "molecule",
            "parent_structure",
        ),
        default=default,
    )


# -----------------------------------------------------------------------------
# Coordinate retrieval
# -----------------------------------------------------------------------------

def get_atom_coordinate(
    atom: AtomLike,
    *,
    scene: bool = False,
    copy: bool = True,
    require_finite: bool = True,
) -> FloatArray:
    """
    Retrieve and validate an atom coordinate.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object, mapping or direct coordinate.
    scene : bool, optional
        Whether scene coordinates should be preferred.
    copy : bool, optional
        Whether a new coordinate array should be returned.
    require_finite : bool, optional
        Whether ``NaN`` and infinite coordinate values should be rejected.

    Returns
    -------
    numpy.ndarray
        Coordinate with shape ``(3,)`` and dtype ``numpy.float64``.

    Raises
    ------
    TypeError
        If no coordinate can be extracted.
    ValueError
        If the coordinate has an invalid shape or non-finite values.
    """

    try:
        return geometry.get_atom_coordinate(
            atom,
            scene=scene,
            copy=copy,
            require_finite=require_finite,
        )

    except (
        AttributeError,
        TypeError,
        ValueError,
    ) as primary_error:
        coordinate_names: Tuple[
            str,
            ...,
        ]

        if scene:
            coordinate_names = (
                "scene_coord",
                "scene_coordinate",
                "scene_coords",
                "coord",
                "coordinate",
                "coords",
                "xyz",
                "position",
            )

        else:
            coordinate_names = (
                "coord",
                "coordinate",
                "coords",
                "xyz",
                "position",
                "scene_coord",
                "scene_coordinate",
            )

        coordinate = _get_object_value(
            atom,
            coordinate_names,
            default=None,
            call_if_callable=True,
        )

        if coordinate is None:
            coordinate = atom

        try:
            return geometry.as_coordinate(
                coordinate,
                scene=scene,
                name="Atom coordinate",
                copy=copy,
                require_finite=require_finite,
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as fallback_error:
            raise TypeError(
                "Could not extract a valid coordinate "
                "from the atom-like object."
            ) from fallback_error


# -----------------------------------------------------------------------------
# Atom identification
# -----------------------------------------------------------------------------

def get_atom_identifier(
    atom: AtomLike,
    *,
    default: str = "",
) -> str:
    """
    Build a human-readable atom identifier.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : str, optional
        Fallback identifier.

    Returns
    -------
    str
        Human-readable identifier.

    Notes
    -----
    The function attempts to include chain, residue and atom information when
    these attributes are available.
    """

    atom_name = get_atom_name(
        atom,
        default="?",
    )

    residue = get_atom_residue(
        atom
    )

    if residue is None:
        return atom_name or default

    residue_name = _normalize_text_value(
        _get_object_value(
            residue,
            (
                "name",
                "residue_name",
                "type",
            ),
            default="UNK",
        ),
        default="UNK",
        uppercase=True,
    )

    residue_number = _get_object_value(
        residue,
        (
            "number",
            "residue_number",
            "position",
            "index",
        ),
        default="?",
    )

    chain_id = _normalize_text_value(
        _get_object_value(
            residue,
            (
                "chain_id",
                "chain",
                "chain_name",
            ),
            default="",
        ),
        default="",
    )

    if chain_id:
        return (
            f"{chain_id}:"
            f"{residue_name}"
            f"{residue_number}:"
            f"{atom_name}"
        )

    return (
        f"{residue_name}"
        f"{residue_number}:"
        f"{atom_name}"
    )


def get_atom_index(
    atom: AtomLike,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Retrieve a stable atom index when available.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object or mapping.
    default : int or None, optional
        Value returned when no index is available.

    Returns
    -------
    int or None
        Atom index.
    """

    value = _get_object_value(
        atom,
        (
            "index",
            "serial_number",
            "serial",
            "atom_index",
        ),
        default=None,
    )

    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return default

    try:
        index = int(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default

    return index


# -----------------------------------------------------------------------------
# Atom validation and classification
# -----------------------------------------------------------------------------

def is_atom_like(
    value: Any,
    *,
    require_coordinate: bool = True,
) -> bool:
    """
    Determine whether a value can be treated as an atom.

    Parameters
    ----------
    value : Any
        Value to inspect.
    require_coordinate : bool, optional
        Whether a valid coordinate is required.

    Returns
    -------
    bool
        ``True`` when the value satisfies the minimum atom interface.
    """

    if value is None:
        return False

    if require_coordinate:
        try:
            get_atom_coordinate(
                value,
                copy=False,
            )

        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            return False

    has_name = bool(
        get_atom_name(
            value
        )
    )

    has_element = bool(
        get_atom_element(
            value,
            infer_from_name=True,
        )
    )

    return bool(
        has_name
        or has_element
        or require_coordinate
    )


def validate_atom(
    atom: AtomLike,
    *,
    require_name: bool = False,
    require_element: bool = False,
    require_residue: bool = False,
    require_structure: bool = False,
    require_coordinate: bool = True,
) -> AtomLike:
    """
    Validate an atom-like object.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object to validate.
    require_name : bool, optional
        Whether the atom must have a non-empty name.
    require_element : bool, optional
        Whether the atom must have an identifiable element.
    require_residue : bool, optional
        Whether the atom must belong to a residue.
    require_structure : bool, optional
        Whether the atom must belong to a structure.
    require_coordinate : bool, optional
        Whether the atom must provide a valid coordinate.

    Returns
    -------
    AtomLike
        Original validated object.

    Raises
    ------
    TypeError
        If ``atom`` is ``None`` or cannot be treated as an atom.
    ValueError
        If a required atom attribute is unavailable.
    """

    if atom is None:
        raise TypeError(
            "atom cannot be None."
        )

    if require_coordinate:
        get_atom_coordinate(
            atom,
            copy=False,
        )

    if (
        require_name
        and not get_atom_name(
            atom
        )
    ):
        raise ValueError(
            "Atom name is required but unavailable."
        )

    if (
        require_element
        and not get_atom_element(
            atom,
            infer_from_name=True,
        )
    ):
        raise ValueError(
            "Atom element is required but unavailable."
        )

    if (
        require_residue
        and get_atom_residue(
            atom
        )
        is None
    ):
        raise ValueError(
            "Atom residue is required but unavailable."
        )

    if (
        require_structure
        and get_atom_structure(
            atom
        )
        is None
    ):
        raise ValueError(
            "Atom structure is required but unavailable."
        )

    return atom


def validate_atom_collection(
    atoms: Iterable[AtomLike],
    *,
    allow_empty: bool = False,
    ignore_none: bool = False,
    require_name: bool = False,
    require_element: bool = False,
    require_residue: bool = False,
    require_structure: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Validate and normalize a collection of atom-like objects.

    Parameters
    ----------
    atoms : iterable of AtomLike
        Atom collection.
    allow_empty : bool, optional
        Whether an empty collection is accepted.
    ignore_none : bool, optional
        Whether ``None`` values should be skipped.
    require_name : bool, optional
        Whether every atom must have a name.
    require_element : bool, optional
        Whether every atom must have an identifiable element.
    require_residue : bool, optional
        Whether every atom must belong to a residue.
    require_structure : bool, optional
        Whether every atom must belong to a structure.
    require_coordinate : bool, optional
        Whether every atom must provide a valid coordinate.

    Returns
    -------
    tuple of AtomLike
        Validated immutable atom collection.

    Raises
    ------
    TypeError
        If the input is not iterable or contains invalid atoms.
    ValueError
        If the collection is empty when ``allow_empty=False`` or a required
        atom attribute is unavailable.
    """

    if atoms is None:
        raise TypeError(
            "atoms cannot be None."
        )

    if isinstance(
        atoms,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "atoms must be an iterable of atom-like objects, "
            "not a string."
        )

    try:
        atom_values = tuple(
            atoms
        )

    except TypeError as error:
        raise TypeError(
            "atoms must be iterable."
        ) from error

    normalized_atoms: List[
        AtomLike
    ] = []

    for index, atom in enumerate(
        atom_values
    ):
        if atom is None:
            if ignore_none:
                continue

            raise TypeError(
                f"Atom at collection index {index} is None."
            )

        try:
            validated_atom = validate_atom(
                atom,
                require_name=require_name,
                require_element=require_element,
                require_residue=require_residue,
                require_structure=require_structure,
                require_coordinate=require_coordinate,
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise type(
                error
            )(
                f"Invalid atom at collection index "
                f"{index}: {error}"
            ) from error

        normalized_atoms.append(
            validated_atom
        )

    if (
        not normalized_atoms
        and not allow_empty
    ):
        raise ValueError(
            "Atom collection cannot be empty."
        )

    return tuple(
        normalized_atoms
    )


def _coordinates_from_validated_atoms(
    atoms: Sequence[AtomLike],
    *,
    scene: bool,
    allow_empty: bool,
    require_finite: bool,
) -> FloatArray:
    """Extract coordinates from an already normalized atom sequence."""

    coordinates: List[FloatArray] = []
    for index, atom in enumerate(atoms):
        try:
            coordinate = get_atom_coordinate(
                atom,
                scene=scene,
                copy=False,
                require_finite=require_finite,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise type(error)(
                f"Invalid atom at collection index {index}: {error}"
            ) from error
        coordinates.append(coordinate)

    if not coordinates:
        return np.empty((0, 3), dtype=np.float64)

    return geometry.as_coordinate_matrix(
        coordinates,
        allow_empty=allow_empty,
        require_finite=require_finite,
        copy=True,
        name="Atom coordinates",
    )


def atom_coordinates(
    atoms: Iterable[AtomLike],
    *,
    scene: bool = False,
    allow_empty: bool = False,
    ignore_none: bool = False,
    require_finite: bool = True,
) -> FloatArray:
    """
    
        Extract coordinates from an atom collection.
    
        Parameters
        ----------
        atoms : iterable of AtomLike
            Atom collection.
        scene : bool, optional
            Whether scene coordinates should be preferred.
        allow_empty : bool, optional
            Whether an empty coordinate matrix is accepted.
        ignore_none : bool, optional
            Whether ``None`` atoms should be skipped.
        require_finite : bool, optional
            Whether non-finite coordinate values should be rejected.
    
        Returns
        -------
        numpy.ndarray
            Coordinate matrix with shape ``(N, 3)``.
        
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=allow_empty,
        ignore_none=ignore_none,
        require_coordinate=False,
    )
    return _coordinates_from_validated_atoms(
        validated_atoms,
        scene=scene,
        allow_empty=allow_empty,
        require_finite=require_finite,
    )

def is_hydrogen_atom(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom is hydrogen.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object.

    Returns
    -------
    bool
        ``True`` for hydrogen atoms.
    """

    atomic_number = get_atom_atomic_number(
        atom
    )

    if atomic_number is not None:
        return atomic_number == 1

    return get_atom_element(
        atom
    ) == "H"


def is_heavy_atom(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom is a non-hydrogen atom.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object.

    Returns
    -------
    bool
        ``True`` for non-hydrogen atoms.
    """

    element = get_atom_element(
        atom
    )

    return bool(
        element
        and element != "H"
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_4_PUBLIC_NAMES = [
    "get_atom_name",
    "infer_element_from_atom_name",
    "get_atom_element",
    "get_atom_atomic_number",
    "get_atom_residue",
    "get_atom_structure",
    "get_atom_coordinate",
    "get_atom_identifier",
    "get_atom_index",
    "is_atom_like",
    "validate_atom",
    "validate_atom_collection",
    "atom_coordinates",
    "is_hydrogen_atom",
    "is_heavy_atom",
]

_register_public_names(_SECTION_4_PUBLIC_NAMES)


# =============================================================================
# End of Section 4
# =============================================================================


# =============================================================================
# Section 5 — Atom collection selection and filtering
# =============================================================================


# -----------------------------------------------------------------------------
# Internal residue and atom helpers
# -----------------------------------------------------------------------------

def _get_residue_name(
    residue: Optional[ResidueLike],
    *,
    default: str = "",
) -> str:
    """Retrieve a normalized residue name."""

    if residue is None:
        return _normalize_text_value(
            default,
            uppercase=True,
        )

    value = _get_object_value(
        residue,
        (
            "name",
            "residue_name",
            "type",
            "resname",
        ),
        default=default,
    )

    return _normalize_text_value(
        value,
        default=default,
        uppercase=True,
    )


def _get_residue_number(
    residue: Optional[ResidueLike],
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """Retrieve a residue number safely."""

    if residue is None:
        return default

    value = _get_object_value(
        residue,
        (
            "number",
            "residue_number",
            "position",
            "index",
            "seq_id",
        ),
        default=None,
    )

    if value is None or isinstance(
        value,
        bool,
    ):
        return default

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def _get_residue_chain_id(
    residue: Optional[ResidueLike],
    *,
    default: str = "",
) -> str:
    """Retrieve a residue chain identifier safely."""

    if residue is None:
        return _normalize_text_value(
            default
        )

    value = _get_object_value(
        residue,
        (
            "chain_id",
            "chain_name",
            "chain",
        ),
        default=default,
    )

    if not isinstance(
        value,
        str,
    ):
        nested_chain_id = _get_object_value(
            value,
            (
                "chain_id",
                "id",
                "name",
            ),
            default=None,
        )

        if nested_chain_id is not None:
            value = nested_chain_id

    return _normalize_text_value(
        value,
        default=default,
    )


def get_residue_contact_key(
    residue: Optional[ResidueLike],
    *,
    default_name: str = "UNK",
    default_number: int = -1,
    default_chain: str = "",
) -> ResidueContactKey:
    """
    Build a stable residue-contact key.

    Parameters
    ----------
    residue : ResidueLike or None
        Residue-like object or mapping.
    default_name : str, optional
        Fallback residue name.
    default_number : int, optional
        Fallback residue number.
    default_chain : str, optional
        Fallback chain identifier.

    Returns
    -------
    ResidueContactKey
        Tuple containing residue name, number and chain identifier.
    """

    residue_name = _get_residue_name(
        residue,
        default=default_name,
    )

    residue_number = _get_residue_number(
        residue,
        default=default_number,
    )

    chain_id = _get_residue_chain_id(
        residue,
        default=default_chain,
    )

    return (
        residue_name,
        (
            default_number
            if residue_number is None
            else residue_number
        ),
        chain_id,
    )


def atoms_share_residue(
    atom_1: AtomLike,
    atom_2: AtomLike,
) -> bool:
    """
    Determine whether two atoms belong to the same residue.

    Parameters
    ----------
    atom_1 : AtomLike
        First atom-like object.
    atom_2 : AtomLike
        Second atom-like object.

    Returns
    -------
    bool
        ``True`` when both atoms belong to the same residue.

    Notes
    -----
    Object identity is preferred. Stable residue keys are used as a fallback
    for synthetic objects and separately materialized residue instances.
    """

    residue_1 = get_atom_residue(
        atom_1
    )

    residue_2 = get_atom_residue(
        atom_2
    )

    if (
        residue_1 is None
        or residue_2 is None
    ):
        return False

    if residue_1 is residue_2:
        return True

    key_1 = get_residue_contact_key(
        residue_1
    )

    key_2 = get_residue_contact_key(
        residue_2
    )

    if key_1 != key_2:
        return False

    structure_1 = get_atom_structure(
        atom_1
    )

    structure_2 = get_atom_structure(
        atom_2
    )

    if (
        structure_1 is not None
        and structure_2 is not None
        and structure_1 is not structure_2
    ):
        return False

    return True


# -----------------------------------------------------------------------------
# Solvent and ion identification
# -----------------------------------------------------------------------------

_FALLBACK_SOLVENT_RESIDUE_NAMES = frozenset(
    {
        "HOH",
        "WAT",
        "H2O",
        "DOD",
        "TIP",
        "TIP3",
        "TIP3P",
        "TIP4",
        "TIP4P",
        "TIP5",
        "TIP5P",
        "SOL",
        "SPC",
        "SPCE",
    }
)

_FALLBACK_ION_RESIDUE_NAMES = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
        "AL",
        "CL",
        "BR",
        "IOD",
        "I",
        "F",
    }
)

_FALLBACK_ION_ELEMENTS = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
        "AL",
        "CL",
        "BR",
        "I",
        "F",
    }
)


def _get_configured_name_set(
    candidate_names: Sequence[str],
    fallback: Iterable[str],
) -> frozenset[str]:
    """Retrieve a normalized name set from ``config``."""

    configured_value: Any = None

    for attribute_name in candidate_names:
        if hasattr(
            config,
            attribute_name,
        ):
            configured_value = getattr(
                config,
                attribute_name,
            )

            if configured_value is not None:
                break

    if configured_value is None:
        configured_value = fallback

    if isinstance(
        configured_value,
        str,
    ):
        configured_values = (
            configured_value,
        )

    else:
        try:
            configured_values = tuple(
                configured_value
            )

        except TypeError:
            configured_values = tuple(
                fallback
            )

    return frozenset(
        _normalize_text_value(
            value,
            uppercase=True,
        )
        for value in configured_values
        if _normalize_text_value(
            value,
            uppercase=True,
        )
    )


def get_solvent_residue_names() -> frozenset[str]:
    """
    Return residue names interpreted as solvent.

    Returns
    -------
    frozenset of str
        Uppercase solvent residue names.
    """

    return _get_configured_name_set(
        (
            "SOLVENT_RESIDUE_NAMES",
            "WATER_RESIDUE_NAMES",
            "DEFAULT_SOLVENT_RESIDUE_NAMES",
        ),
        _FALLBACK_SOLVENT_RESIDUE_NAMES,
    )


def get_ion_residue_names() -> frozenset[str]:
    """
    Return residue names interpreted as ions.

    Returns
    -------
    frozenset of str
        Uppercase ion residue names.
    """

    return _get_configured_name_set(
        (
            "ION_RESIDUE_NAMES",
            "DEFAULT_ION_RESIDUE_NAMES",
        ),
        _FALLBACK_ION_RESIDUE_NAMES,
    )


def get_ion_elements() -> frozenset[str]:
    """
    Return element symbols commonly interpreted as monatomic ions.

    Returns
    -------
    frozenset of str
        Uppercase element symbols.
    """

    return _get_configured_name_set(
        (
            "ION_ELEMENTS",
            "DEFAULT_ION_ELEMENTS",
        ),
        _FALLBACK_ION_ELEMENTS,
    )


def is_solvent_atom(
    atom: AtomLike,
    *,
    solvent_residue_names: Optional[
        Iterable[str]
    ] = None,
) -> bool:
    """
    Determine whether an atom belongs to a solvent residue.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object.
    solvent_residue_names : iterable of str or None, optional
        Custom solvent residue names. Configured defaults are used when
        omitted.

    Returns
    -------
    bool
        ``True`` when the atom belongs to a recognized solvent residue.
    """

    residue = get_atom_residue(
        atom
    )

    residue_name = _get_residue_name(
        residue
    )

    if not residue_name:
        return False

    if solvent_residue_names is None:
        normalized_names = (
            get_solvent_residue_names()
        )

    else:
        normalized_names = frozenset(
            _normalize_text_value(
                name,
                uppercase=True,
            )
            for name in solvent_residue_names
        )

    return residue_name in normalized_names


def is_ion_atom(
    atom: AtomLike,
    *,
    ion_residue_names: Optional[
        Iterable[str]
    ] = None,
    ion_elements: Optional[
        Iterable[str]
    ] = None,
) -> bool:
    """
    Determine whether an atom represents an ion.

    Parameters
    ----------
    atom : AtomLike
        Atom-like object.
    ion_residue_names : iterable of str or None, optional
        Custom ion residue names.
    ion_elements : iterable of str or None, optional
        Custom element symbols considered ionic.

    Returns
    -------
    bool
        ``True`` when the atom is interpreted as an ion.

    Notes
    -----
    Residue names are checked first. Element-based classification is applied
    primarily to single-atom residues to avoid classifying elements such as
    calcium inside larger synthetic structures as free ions.
    """

    residue = get_atom_residue(
        atom
    )

    residue_name = _get_residue_name(
        residue
    )

    if ion_residue_names is None:
        normalized_residue_names = (
            get_ion_residue_names()
        )

    else:
        normalized_residue_names = frozenset(
            _normalize_text_value(
                name,
                uppercase=True,
            )
            for name in ion_residue_names
        )

    if (
        residue_name
        and residue_name
        in normalized_residue_names
    ):
        return True

    element = get_atom_element(
        atom
    )

    if ion_elements is None:
        normalized_ion_elements = (
            get_ion_elements()
        )

    else:
        normalized_ion_elements = frozenset(
            _normalize_text_value(
                value,
                uppercase=True,
            )
            for value in ion_elements
        )

    if element not in normalized_ion_elements:
        return False

    if residue is None:
        return True

    residue_atoms = _get_object_value(
        residue,
        (
            "atoms",
            "atom_collection",
        ),
        default=None,
    )

    if residue_atoms is None:
        return True

    try:
        return len(
            residue_atoms
        ) == 1

    except TypeError:
        try:
            return len(
                tuple(
                    residue_atoms
                )
            ) == 1

        except TypeError:
            return True


# -----------------------------------------------------------------------------
# Generic atom filtering
# -----------------------------------------------------------------------------

def filter_atoms(
    atoms: Iterable[AtomLike],
    *,
    predicate: Optional[
        Callable[
            [AtomLike],
            bool,
        ]
    ] = None,
    exclude_solvent: bool = False,
    exclude_ions: bool = False,
    exclude_hydrogens: bool = False,
    include_only_hydrogens: bool = False,
    solvent_residue_names: Optional[
        Iterable[str]
    ] = None,
    ion_residue_names: Optional[
        Iterable[str]
    ] = None,
    ion_elements: Optional[
        Iterable[str]
    ] = None,
    allow_empty: bool = True,
    ignore_none: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Filter an atom collection.

    Parameters
    ----------
    atoms : iterable of AtomLike
        Source atom collection.
    predicate : callable or None, optional
        Additional predicate. Atoms are retained when it returns ``True``.
    exclude_solvent : bool, optional
        Whether atoms belonging to solvent residues should be removed.
    exclude_ions : bool, optional
        Whether ionic atoms should be removed.
    exclude_hydrogens : bool, optional
        Whether hydrogen atoms should be removed.
    include_only_hydrogens : bool, optional
        Whether only hydrogen atoms should be retained.
    solvent_residue_names : iterable of str or None, optional
        Custom solvent residue names.
    ion_residue_names : iterable of str or None, optional
        Custom ion residue names.
    ion_elements : iterable of str or None, optional
        Custom ionic element symbols.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    ignore_none : bool, optional
        Whether ``None`` entries should be skipped.
    require_coordinate : bool, optional
        Whether every retained atom must provide a valid coordinate.

    Returns
    -------
    tuple of AtomLike
        Filtered immutable atom collection.

    Raises
    ------
    ValueError
        If contradictory hydrogen filters are requested or the resulting
        collection is empty when ``allow_empty=False``.
    """

    if (
        exclude_hydrogens
        and include_only_hydrogens
    ):
        raise ValueError(
            "exclude_hydrogens and "
            "include_only_hydrogens cannot both be True."
        )

    normalized_atoms = (
        validate_atom_collection(
            atoms,
            allow_empty=allow_empty,
            ignore_none=ignore_none,
            require_coordinate=(
                require_coordinate
            ),
        )
    )

    selected_atoms: List[
        AtomLike
    ] = []

    for atom in normalized_atoms:
        hydrogen = is_hydrogen_atom(
            atom
        )

        if (
            exclude_hydrogens
            and hydrogen
        ):
            continue

        if (
            include_only_hydrogens
            and not hydrogen
        ):
            continue

        if (
            exclude_solvent
            and is_solvent_atom(
                atom,
                solvent_residue_names=(
                    solvent_residue_names
                ),
            )
        ):
            continue

        if (
            exclude_ions
            and is_ion_atom(
                atom,
                ion_residue_names=(
                    ion_residue_names
                ),
                ion_elements=(
                    ion_elements
                ),
            )
        ):
            continue

        if (
            predicate is not None
            and not bool(
                predicate(
                    atom
                )
            )
        ):
            continue

        selected_atoms.append(
            atom
        )

    if (
        not selected_atoms
        and not allow_empty
    ):
        raise ValueError(
            "Atom filtering produced an empty collection."
        )

    return tuple(
        selected_atoms
    )


def select_heavy_atoms(
    atoms: Iterable[AtomLike],
    *,
    exclude_solvent: bool = False,
    exclude_ions: bool = False,
    allow_empty: bool = True,
    ignore_none: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Select non-hydrogen atoms.

    Parameters
    ----------
    atoms : iterable of AtomLike
        Source atom collection.
    exclude_solvent : bool, optional
        Whether solvent atoms should be excluded.
    exclude_ions : bool, optional
        Whether ionic atoms should be excluded.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    ignore_none : bool, optional
        Whether ``None`` entries should be skipped.
    require_coordinate : bool, optional
        Whether valid coordinates are required.

    Returns
    -------
    tuple of AtomLike
        Heavy-atom collection.
    """

    return filter_atoms(
        atoms,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        exclude_hydrogens=True,
        allow_empty=allow_empty,
        ignore_none=ignore_none,
        require_coordinate=require_coordinate,
    )


def select_hydrogen_atoms(
    atoms: Iterable[AtomLike],
    *,
    exclude_solvent: bool = False,
    exclude_ions: bool = False,
    allow_empty: bool = True,
    ignore_none: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Select hydrogen atoms.

    Parameters
    ----------
    atoms : iterable of AtomLike
        Source atom collection.
    exclude_solvent : bool, optional
        Whether solvent hydrogens should be excluded.
    exclude_ions : bool, optional
        Whether ionic entries should be excluded.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    ignore_none : bool, optional
        Whether ``None`` entries should be skipped.
    require_coordinate : bool, optional
        Whether valid coordinates are required.

    Returns
    -------
    tuple of AtomLike
        Hydrogen-atom collection.
    """

    return filter_atoms(
        atoms,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        include_only_hydrogens=True,
        allow_empty=allow_empty,
        ignore_none=ignore_none,
        require_coordinate=require_coordinate,
    )


# -----------------------------------------------------------------------------
# Model and structure atom extraction
# -----------------------------------------------------------------------------

def get_structure_atoms(
    structure: StructureLike,
    *,
    allow_empty: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Extract atoms from a structure-like object.

    Parameters
    ----------
    structure : StructureLike
        ChimeraX structure, synthetic structure or atom iterable.
    allow_empty : bool, optional
        Whether an empty collection is accepted.
    require_coordinate : bool, optional
        Whether atoms must provide valid coordinates.

    Returns
    -------
    tuple of AtomLike
        Extracted atoms.

    Raises
    ------
    TypeError
        If no atom collection can be obtained.
    """

    if structure is None:
        raise TypeError(
            "structure cannot be None."
        )

    atom_collection = _get_object_value(
        structure,
        (
            "atoms",
            "atom_collection",
            "all_atoms",
        ),
        default=None,
        call_if_callable=True,
    )

    if atom_collection is None:
        if isinstance(
            structure,
            (
                str,
                bytes,
                Mapping,
            ),
        ):
            raise TypeError(
                "Could not extract atoms from the "
                "structure-like object."
            )

        try:
            atom_collection = tuple(
                structure
            )

        except TypeError as error:
            raise TypeError(
                "Could not extract atoms from the "
                "structure-like object."
            ) from error

    return validate_atom_collection(
        atom_collection,
        allow_empty=allow_empty,
        require_coordinate=require_coordinate,
    )


def select_ligand_atoms(
    ligand: Union[
        LigandLike,
        Iterable[AtomLike],
    ],
    *,
    heavy_only: bool = False,
    exclude_solvent: bool = True,
    exclude_ions: bool = False,
    allow_empty: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Select atoms belonging to a ligand.

    Parameters
    ----------
    ligand : LigandLike or iterable of AtomLike
        Ligand structure, residue or explicit atom collection.
    heavy_only : bool, optional
        Whether hydrogen atoms should be excluded.
    exclude_solvent : bool, optional
        Whether solvent atoms should be excluded.
    exclude_ions : bool, optional
        Whether ionic atoms should be excluded.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    require_coordinate : bool, optional
        Whether atoms must provide valid coordinates.

    Returns
    -------
    tuple of AtomLike
        Selected ligand atoms.
    """

    ligand_atoms = get_structure_atoms(
        ligand,
        allow_empty=allow_empty,
        require_coordinate=require_coordinate,
    )

    return filter_atoms(
        ligand_atoms,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        exclude_hydrogens=heavy_only,
        allow_empty=allow_empty,
        require_coordinate=require_coordinate,
    )


def select_receptor_atoms(
    receptor: Union[
        ReceptorLike,
        Iterable[AtomLike],
    ],
    *,
    heavy_only: bool = False,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    allow_empty: bool = False,
    require_coordinate: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Select atoms belonging to a receptor.

    Parameters
    ----------
    receptor : ReceptorLike or iterable of AtomLike
        Receptor structure or explicit atom collection.
    heavy_only : bool, optional
        Whether hydrogen atoms should be excluded.
    exclude_solvent : bool, optional
        Whether solvent atoms should be excluded.
    exclude_ions : bool, optional
        Whether ions should be excluded.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    require_coordinate : bool, optional
        Whether atoms must provide valid coordinates.

    Returns
    -------
    tuple of AtomLike
        Selected receptor atoms.
    """

    receptor_atoms = get_structure_atoms(
        receptor,
        allow_empty=allow_empty,
        require_coordinate=require_coordinate,
    )

    return filter_atoms(
        receptor_atoms,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        exclude_hydrogens=heavy_only,
        allow_empty=allow_empty,
        require_coordinate=require_coordinate,
    )


# -----------------------------------------------------------------------------
# Pairwise collection filtering
# -----------------------------------------------------------------------------

def exclude_same_residue_pairs(
    atom_pairs: Iterable[ContactPair],
) -> Tuple[ContactPair, ...]:
    """
    Remove atom pairs belonging to the same residue.

    Parameters
    ----------
    atom_pairs : iterable of ContactPair
        Atom pairs to filter.

    Returns
    -------
    tuple of ContactPair
        Pairs whose atoms do not share a residue.
    """

    if atom_pairs is None:
        raise TypeError(
            "atom_pairs cannot be None."
        )

    selected_pairs: List[
        ContactPair
    ] = []

    for pair_index, pair in enumerate(
        atom_pairs
    ):
        try:
            atom_1, atom_2 = pair

        except (
            TypeError,
            ValueError,
        ) as error:
            raise TypeError(
                "Each contact pair must contain exactly "
                f"two atoms. Invalid pair index: {pair_index}."
            ) from error

        if atoms_share_residue(
            atom_1,
            atom_2,
        ):
            continue

        selected_pairs.append(
            (
                atom_1,
                atom_2,
            )
        )

    return tuple(
        selected_pairs
    )


def select_contact_collections(
    ligand: Union[
        LigandLike,
        Iterable[AtomLike],
    ],
    receptor: Union[
        ReceptorLike,
        Iterable[AtomLike],
    ],
    *,
    heavy_only: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    require_coordinate: bool = True,
) -> Tuple[
    Tuple[AtomLike, ...],
    Tuple[AtomLike, ...],
]:
    """
    Prepare ligand and receptor atom collections for contact analysis.

    Parameters
    ----------
    ligand : LigandLike or iterable of AtomLike
        Ligand source.
    receptor : ReceptorLike or iterable of AtomLike
        Receptor source.
    heavy_only : bool, optional
        Whether both collections should contain only heavy atoms.
    exclude_solvent : bool, optional
        Whether solvent atoms should be excluded.
    exclude_ions : bool, optional
        Whether ions should be excluded from the receptor collection.
    require_coordinate : bool, optional
        Whether valid coordinates are required.

    Returns
    -------
    tuple
        ``(ligand_atoms, receptor_atoms)``.

    Notes
    -----
    Ions are not excluded from the ligand collection by default because a
    ligand may itself be ionic or contain a coordinated metal.
    """

    ligand_atoms = select_ligand_atoms(
        ligand,
        heavy_only=heavy_only,
        exclude_solvent=exclude_solvent,
        exclude_ions=False,
        allow_empty=False,
        require_coordinate=require_coordinate,
    )

    receptor_atoms = select_receptor_atoms(
        receptor,
        heavy_only=heavy_only,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        allow_empty=False,
        require_coordinate=require_coordinate,
    )

    return (
        ligand_atoms,
        receptor_atoms,
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES = [
    "get_residue_contact_key",
    "atoms_share_residue",
    "get_solvent_residue_names",
    "get_ion_residue_names",
    "get_ion_elements",
    "is_solvent_atom",
    "is_ion_atom",
    "filter_atoms",
    "select_heavy_atoms",
    "select_hydrogen_atoms",
    "get_structure_atoms",
    "select_ligand_atoms",
    "select_receptor_atoms",
    "exclude_same_residue_pairs",
    "select_contact_collections",
]

_register_public_names(_SECTION_5_PUBLIC_NAMES)


# =============================================================================
# End of Section 5
# =============================================================================


# =============================================================================
# Section 6 — Contact search
# =============================================================================


# -----------------------------------------------------------------------------
# Internal validation helpers
# -----------------------------------------------------------------------------

def _validate_scene_flag(
    scene: bool,
) -> bool:
    """Validate whether scene-transformed coordinates should be used."""

    if not isinstance(
        scene,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "scene must be a boolean value."
        )

    return bool(
        scene
    )


def _validate_contact_cutoff(
    cutoff: Number,
    *,
    name: str = "cutoff",
    allow_zero: bool = False,
) -> np.float64:
    """Validate a contact-distance cutoff."""

    if isinstance(
        cutoff,
        bool,
    ) or not isinstance(
        cutoff,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        raise TypeError(
            f"{name} must be a numeric value."
        )

    cutoff_value = np.float64(
        cutoff
    )

    if not np.isfinite(
        cutoff_value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    minimum_value = (
        np.float64(0.0)
        if allow_zero
        else np.nextafter(
            np.float64(0.0),
            np.float64(1.0),
        )
    )

    if cutoff_value < minimum_value:
        comparison = (
            "greater than or equal to zero"
            if allow_zero
            else "greater than zero"
        )

        raise ValueError(
            f"{name} must be {comparison}."
        )

    return cutoff_value


def _validate_block_size(
    block_size: Optional[int],
) -> Optional[int]:
    """Validate a distance-matrix block size."""

    if block_size is None:
        return None

    if isinstance(
        block_size,
        bool,
    ) or not isinstance(
        block_size,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "block_size must be an integer or None."
        )

    normalized_block_size = int(
        block_size
    )

    if normalized_block_size <= 0:
        raise ValueError(
            "block_size must be greater than zero."
        )

    return normalized_block_size


def _validate_maximum_matrix_elements(
    maximum_matrix_elements: int,
) -> int:
    """Validate the maximum number of distance-matrix elements."""

    if isinstance(
        maximum_matrix_elements,
        bool,
    ) or not isinstance(
        maximum_matrix_elements,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "maximum_matrix_elements must be an integer."
        )

    normalized_limit = int(
        maximum_matrix_elements
    )

    if normalized_limit <= 0:
        raise ValueError(
            "maximum_matrix_elements must be greater than zero."
        )

    return normalized_limit


def _get_default_contact_cutoff() -> np.float64:
    """Retrieve the default contact cutoff from ``config``."""

    candidate_names = (
        "DEFAULT_CONTACT_DISTANCE",
        "DEFAULT_CONTACT_CUTOFF",
        "CONTACT_DISTANCE_CUTOFF",
        "CONTACT_CUTOFF",
    )

    for attribute_name in candidate_names:
        if hasattr(
            config,
            attribute_name,
        ):
            value = getattr(
                config,
                attribute_name,
            )

            if value is not None:
                return _validate_contact_cutoff(
                    value,
                    name=(
                        f"config."
                        f"{attribute_name}"
                    ),
                )

    return np.float64(
        4.0
    )


def _get_default_maximum_matrix_elements() -> int:
    """Retrieve the default distance-matrix size limit."""

    candidate_names = (
        "MAXIMUM_CONTACT_MATRIX_ELEMENTS",
        "MAX_CONTACT_MATRIX_ELEMENTS",
        "CONTACT_MATRIX_ELEMENT_LIMIT",
    )

    for attribute_name in candidate_names:
        if hasattr(
            config,
            attribute_name,
        ):
            value = getattr(
                config,
                attribute_name,
            )

            if value is not None:
                return (
                    _validate_maximum_matrix_elements(
                        value
                    )
                )

    return 4_000_000


def _resolve_contact_cutoff(
    cutoff: Optional[Number],
) -> np.float64:
    """Resolve an explicit or configured contact cutoff."""

    if cutoff is None:
        return _get_default_contact_cutoff()

    return _validate_contact_cutoff(
        cutoff
    )


def _resolve_matrix_element_limit(value: Optional[int]) -> int:
    """Resolve an explicit or configured distance-matrix element limit."""

    if value is None:
        return _get_default_maximum_matrix_elements()
    return _validate_maximum_matrix_elements(value)


def _validate_optional_limit(
    value: Optional[int],
    *,
    name: str,
    allow_zero: bool = True,
) -> Optional[int]:
    """Validate an optional non-negative or positive integer limit."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer or None.")
    normalized = int(value)
    minimum = 0 if allow_zero else 1
    if normalized < minimum:
        if allow_zero:
            raise ValueError(f"{name} cannot be negative.")
        raise ValueError(f"{name} must be greater than zero.")
    return normalized


# -----------------------------------------------------------------------------
# Distance strategy helpers
# -----------------------------------------------------------------------------



def _calculate_squared_distance_block(
    coordinates_1: FloatArray,
    coordinates_2: FloatArray,
) -> FloatArray:
    """Calculate a block of squared pairwise distances."""

    displacement = (
        coordinates_1[
            :,
            np.newaxis,
            :,
        ]
        - coordinates_2[
            np.newaxis,
            :,
            :,
        ]
    )

    squared_distances = np.einsum(
        "ijk,ijk->ij",
        displacement,
        displacement,
        dtype=np.float64,
    )

    np.maximum(
        squared_distances,
        np.float64(0.0),
        out=squared_distances,
    )

    return squared_distances


def _resolve_contact_block_size(
    atom_count_1: int,
    atom_count_2: int,
    *,
    block_size: Optional[int],
    maximum_matrix_elements: int,
) -> Optional[int]:
    """Determine whether full-matrix or blocked processing should be used."""

    validated_block_size = (
        _validate_block_size(
            block_size
        )
    )

    if validated_block_size is not None:
        return validated_block_size

    matrix_elements = (
        int(
            atom_count_1
        )
        * int(
            atom_count_2
        )
    )

    if (
        matrix_elements
        <= maximum_matrix_elements
    ):
        return None

    automatic_block_size = (
        maximum_matrix_elements
        // max(
            atom_count_2,
            1,
        )
    )

    return max(
        1,
        int(
            automatic_block_size
        ),
    )


def _iter_coordinate_blocks(
    coordinates: FloatArray,
    block_size: int,
) -> Iterator[
    Tuple[
        int,
        int,
        FloatArray,
    ]
]:
    """Yield coordinate blocks and their index limits."""

    point_count = int(
        coordinates.shape[
            0
        ]
    )

    for start in range(
        0,
        point_count,
        block_size,
    ):
        stop = min(
            start + block_size,
            point_count,
        )

        yield (
            start,
            stop,
            coordinates[
                start:stop
            ],
        )


# -----------------------------------------------------------------------------
# Contact result construction
# -----------------------------------------------------------------------------

def _build_atom_contact(
    atom_1: AtomLike,
    atom_2: AtomLike,
    *,
    distance: Number,
    cutoff: Number,
    atom_1_index: Optional[int] = None,
    atom_2_index: Optional[int] = None,
    classification: ContactClassification = CONTACT_TYPE_CONTACT,
    metadata: Optional[Mapping[str, Any]] = None,
    scene: bool = True,
    coordinate_1: Optional[FloatArray] = None,
    coordinate_2: Optional[FloatArray] = None,
) -> AtomContact:
    """Build an atom contact, reusing validated coordinates when supplied."""

    scene_value = _validate_scene_flag(scene)
    if coordinate_1 is None:
        coordinate_1 = get_atom_coordinate(atom_1, scene=scene_value, copy=True)
    else:
        coordinate_1 = np.array(coordinate_1, dtype=np.float64, copy=True)
    if coordinate_2 is None:
        coordinate_2 = get_atom_coordinate(atom_2, scene=scene_value, copy=True)
    else:
        coordinate_2 = np.array(coordinate_2, dtype=np.float64, copy=True)

    distance_value = np.float64(distance)
    cutoff_value = np.float64(cutoff)
    geometry_result = ContactGeometry(
        atom_1=atom_1,
        atom_2=atom_2,
        coordinate_1=coordinate_1,
        coordinate_2=coordinate_2,
        distance=distance_value,
        cutoff=cutoff_value,
        contact_compatible=bool(distance_value <= cutoff_value),
        index_1=atom_1_index,
        index_2=atom_2_index,
        metadata={
            "source": "contacts.find_atom_contacts",
            "scene_coordinates": scene_value,
        },
    )
    return AtomContact(
        atom_1=atom_1,
        atom_2=atom_2,
        geometry=geometry_result,
        classification=classification,
        atom_1_index=atom_1_index,
        atom_2_index=atom_2_index,
        residue_1=get_atom_residue(atom_1),
        residue_2=get_atom_residue(atom_2),
        metadata={} if metadata is None else dict(metadata),
    )

def _contact_sort_key(
    contact: AtomContact,
) -> Tuple[
    np.float64,
    int,
    int,
]:
    """Build a deterministic sorting key for contacts."""

    index_1 = (
        contact.atom_1_index
        if contact.atom_1_index
        is not None
        else -1
    )

    index_2 = (
        contact.atom_2_index
        if contact.atom_2_index
        is not None
        else -1
    )

    return (
        contact.distance,
        index_1,
        index_2,
    )


# -----------------------------------------------------------------------------
# Atom-collection contact search
# -----------------------------------------------------------------------------

def find_atom_contacts(
    atoms_1: Iterable[AtomLike],
    atoms_2: Iterable[AtomLike],
    *,
    cutoff: Optional[Number] = None,
    exclude_same_residue: bool = False,
    exclude_identical_atoms: bool = True,
    block_size: Optional[int] = None,
    maximum_matrix_elements: Optional[int] = None,
    sort_by_distance: bool = True,
    maximum_contacts: Optional[int] = None,
    classification: ContactClassification = CONTACT_TYPE_CONTACT,
    metadata: Optional[Mapping[str, Any]] = None,
    scene: bool = True,
) -> Tuple[AtomContact, ...]:
    """
    
        Find contacts between two atom collections.
    
        Parameters
        ----------
        atoms_1 : iterable of AtomLike
            First atom collection, normally ligand atoms.
        atoms_2 : iterable of AtomLike
            Second atom collection, normally receptor atoms.
        cutoff : Number or None, optional
            Maximum contact distance in angstroms. The configured default is
            used when omitted.
        exclude_same_residue : bool, optional
            Whether atom pairs belonging to the same residue should be ignored.
        exclude_identical_atoms : bool, optional
            Whether pairs containing the exact same atom object should be
            ignored.
        block_size : int or None, optional
            Number of atoms from ``atoms_1`` processed per block. When omitted,
            full-matrix or blocked processing is selected automatically.
        maximum_matrix_elements : int or None, optional
            Maximum number of entries allowed in a full distance matrix.
        sort_by_distance : bool, optional
            Whether returned contacts should be ordered by increasing distance.
        maximum_contacts : int or None, optional
            Maximum number of contacts returned after optional sorting.
        classification : str, optional
            Initial classification assigned to every detected contact.
        metadata : mapping or None, optional
            Metadata copied to every contact.
        scene : bool, optional
            Whether ChimeraX scene coordinates should be preferred.
    
        Returns
        -------
        tuple of AtomContact
            Detected atom-level contacts.
    
        Raises
        ------
        TypeError
            If collection or control parameters are invalid.
        ValueError
            If either atom collection is empty or parameter values are invalid.
    
        Notes
        -----
        The cutoff comparison is performed using squared distances. Square roots
        are calculated only for pairs that satisfy the cutoff, reducing work for
        sparse contact searches.
        
    """

    normalized_atoms_1 = validate_atom_collection(
        atoms_1, allow_empty=False, require_coordinate=False
    )
    normalized_atoms_2 = validate_atom_collection(
        atoms_2, allow_empty=False, require_coordinate=False
    )
    scene_value = _validate_scene_flag(scene)
    cutoff_value = _resolve_contact_cutoff(cutoff)
    cutoff_squared = np.float64(cutoff_value * cutoff_value)
    matrix_element_limit = _resolve_matrix_element_limit(maximum_matrix_elements)
    maximum_contacts = _validate_optional_limit(
        maximum_contacts, name="maximum_contacts", allow_zero=True
    )
    if maximum_contacts == 0:
        return ()

    coordinates_1 = _coordinates_from_validated_atoms(
        normalized_atoms_1,
        scene=scene_value,
        allow_empty=False,
        require_finite=True,
    )
    coordinates_2 = _coordinates_from_validated_atoms(
        normalized_atoms_2,
        scene=scene_value,
        allow_empty=False,
        require_finite=True,
    )
    resolved_block_size = _resolve_contact_block_size(
        len(normalized_atoms_1),
        len(normalized_atoms_2),
        block_size=block_size,
        maximum_matrix_elements=matrix_element_limit,
    )

    contact_metadata = {} if metadata is None else dict(metadata)
    contact_metadata.setdefault("cutoff", float(cutoff_value))
    contact_metadata.setdefault("scene_coordinates", scene_value)
    contact_metadata.setdefault(
        "search_strategy", "full_matrix" if resolved_block_size is None else "blocked"
    )
    contacts: List[AtomContact] = []

    def process_distance_block(
        squared_distances: FloatArray,
        *,
        index_offset_1: int,
    ) -> None:
        for local_index_1_raw, index_2_raw in np.argwhere(
            squared_distances <= cutoff_squared
        ):
            local_index_1 = int(local_index_1_raw)
            index_1 = index_offset_1 + local_index_1
            index_2 = int(index_2_raw)
            atom_1 = normalized_atoms_1[index_1]
            atom_2 = normalized_atoms_2[index_2]
            if exclude_identical_atoms and atom_1 is atom_2:
                continue
            if exclude_same_residue and atoms_share_residue(atom_1, atom_2):
                continue
            distance_value = np.float64(
                np.sqrt(squared_distances[local_index_1, index_2])
            )
            contacts.append(
                _build_atom_contact(
                    atom_1,
                    atom_2,
                    distance=distance_value,
                    cutoff=cutoff_value,
                    atom_1_index=index_1,
                    atom_2_index=index_2,
                    classification=classification,
                    metadata=contact_metadata,
                    scene=scene_value,
                    coordinate_1=coordinates_1[index_1],
                    coordinate_2=coordinates_2[index_2],
                )
            )

    if resolved_block_size is None:
        process_distance_block(
            _calculate_squared_distance_block(coordinates_1, coordinates_2),
            index_offset_1=0,
        )
    else:
        for block_start, _, coordinate_block in _iter_coordinate_blocks(
            coordinates_1, resolved_block_size
        ):
            process_distance_block(
                _calculate_squared_distance_block(coordinate_block, coordinates_2),
                index_offset_1=block_start,
            )

    if sort_by_distance:
        contacts.sort(key=_contact_sort_key)
    if maximum_contacts is not None:
        del contacts[maximum_contacts:]
    return tuple(contacts)

# -----------------------------------------------------------------------------
# Closest-contact search
# -----------------------------------------------------------------------------

def _closest_valid_pair_in_block(
    squared_distances: FloatArray,
    atoms_1: Sequence[AtomLike],
    atoms_2: Sequence[AtomLike],
    *,
    index_offset_1: int,
    exclude_same_residue: bool,
    exclude_identical_atoms: bool,
) -> Optional[Tuple[np.float64, int, int]]:
    """Return the closest eligible pair in one distance block."""

    candidates = squared_distances
    while candidates.size:
        flat_index = int(np.argmin(candidates))
        distance_squared = np.float64(candidates.flat[flat_index])
        if not np.isfinite(distance_squared):
            return None
        local_index_1, index_2 = np.unravel_index(flat_index, candidates.shape)
        index_1 = index_offset_1 + int(local_index_1)
        index_2 = int(index_2)
        atom_1 = atoms_1[index_1]
        atom_2 = atoms_2[index_2]
        invalid = (
            (exclude_identical_atoms and atom_1 is atom_2)
            or (exclude_same_residue and atoms_share_residue(atom_1, atom_2))
        )
        if not invalid:
            return distance_squared, index_1, index_2
        if candidates is squared_distances:
            candidates = squared_distances.copy()
        candidates[local_index_1, index_2] = np.inf
    return None


def closest_contact(
    atoms_1: Iterable[AtomLike],
    atoms_2: Iterable[AtomLike],
    *,
    cutoff: Optional[Number] = None,
    exclude_same_residue: bool = False,
    exclude_identical_atoms: bool = True,
    block_size: Optional[int] = None,
    maximum_matrix_elements: Optional[int] = None,
    require_within_cutoff: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    scene: bool = True,
) -> Optional[AtomContact]:
    """
    
        Find the closest valid atom pair between two collections.
    
        Parameters
        ----------
        atoms_1 : iterable of AtomLike
            First atom collection.
        atoms_2 : iterable of AtomLike
            Second atom collection.
        cutoff : Number or None, optional
            Contact cutoff stored in the result.
        exclude_same_residue : bool, optional
            Whether same-residue pairs should be ignored.
        exclude_identical_atoms : bool, optional
            Whether pairs containing the same object should be ignored.
        block_size : int or None, optional
            Block size used for distance processing.
        maximum_matrix_elements : int or None, optional
            Full-matrix element limit.
        require_within_cutoff : bool, optional
            Whether ``None`` should be returned when the closest pair lies
            outside the cutoff.
        metadata : mapping or None, optional
            Additional result metadata.
        scene : bool, optional
            Whether ChimeraX scene coordinates should be preferred.
    
        Returns
        -------
        AtomContact or None
            Closest valid contact, or ``None`` when no eligible pair exists.
    
        Notes
        -----
        Unlike :func:`find_atom_contacts`, this function can return the closest
        pair even when it lies outside the contact cutoff. In that case,
        ``result.is_contact`` is ``False``.
        
    """

    normalized_atoms_1 = validate_atom_collection(
        atoms_1, allow_empty=False, require_coordinate=False
    )
    normalized_atoms_2 = validate_atom_collection(
        atoms_2, allow_empty=False, require_coordinate=False
    )
    scene_value = _validate_scene_flag(scene)
    cutoff_value = _resolve_contact_cutoff(cutoff)
    matrix_element_limit = _resolve_matrix_element_limit(maximum_matrix_elements)
    coordinates_1 = _coordinates_from_validated_atoms(
        normalized_atoms_1,
        scene=scene_value,
        allow_empty=False,
        require_finite=True,
    )
    coordinates_2 = _coordinates_from_validated_atoms(
        normalized_atoms_2,
        scene=scene_value,
        allow_empty=False,
        require_finite=True,
    )
    resolved_block_size = _resolve_contact_block_size(
        len(normalized_atoms_1),
        len(normalized_atoms_2),
        block_size=block_size,
        maximum_matrix_elements=matrix_element_limit,
    )

    best: Optional[Tuple[np.float64, int, int]] = None

    def inspect(squared_distances: FloatArray, *, index_offset_1: int) -> None:
        nonlocal best
        candidate = _closest_valid_pair_in_block(
            squared_distances,
            normalized_atoms_1,
            normalized_atoms_2,
            index_offset_1=index_offset_1,
            exclude_same_residue=exclude_same_residue,
            exclude_identical_atoms=exclude_identical_atoms,
        )
        if candidate is not None and (best is None or candidate[0] < best[0]):
            best = candidate

    if resolved_block_size is None:
        inspect(
            _calculate_squared_distance_block(coordinates_1, coordinates_2),
            index_offset_1=0,
        )
    else:
        for block_start, _, coordinate_block in _iter_coordinate_blocks(
            coordinates_1, resolved_block_size
        ):
            inspect(
                _calculate_squared_distance_block(coordinate_block, coordinates_2),
                index_offset_1=block_start,
            )

    if best is None:
        return None
    best_distance_squared, best_index_1, best_index_2 = best
    distance_value = np.float64(np.sqrt(best_distance_squared))
    if require_within_cutoff and distance_value > cutoff_value:
        return None

    result_metadata = {} if metadata is None else dict(metadata)
    result_metadata.setdefault(
        "search_strategy", "full_matrix" if resolved_block_size is None else "blocked"
    )
    result_metadata.setdefault("closest_pair", True)
    result_metadata.setdefault("scene_coordinates", scene_value)
    return _build_atom_contact(
        normalized_atoms_1[best_index_1],
        normalized_atoms_2[best_index_2],
        distance=distance_value,
        cutoff=cutoff_value,
        atom_1_index=best_index_1,
        atom_2_index=best_index_2,
        classification=(
            CONTACT_TYPE_CONTACT
            if distance_value <= cutoff_value
            else CONTACT_TYPE_UNKNOWN
        ),
        metadata=result_metadata,
        scene=scene_value,
        coordinate_1=coordinates_1[best_index_1],
        coordinate_2=coordinates_2[best_index_2],
    )

# -----------------------------------------------------------------------------
# High-level receptor-ligand contact search
# -----------------------------------------------------------------------------

def find_contacts(
    ligand: Union[LigandLike, Iterable[AtomLike]],
    receptor: Union[ReceptorLike, Iterable[AtomLike]],
    *,
    cutoff: Optional[Number] = None,
    heavy_only: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    exclude_same_residue: bool = True,
    block_size: Optional[int] = None,
    maximum_matrix_elements: Optional[int] = None,
    sort_by_distance: bool = True,
    maximum_contacts: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    scene: bool = True,
) -> Tuple[AtomContact, ...]:
    """
    
        Find contacts between a ligand and receptor.
    
        Parameters
        ----------
        ligand : LigandLike or iterable of AtomLike
            Ligand structure, residue or atom collection.
        receptor : ReceptorLike or iterable of AtomLike
            Receptor structure or atom collection.
        cutoff : Number or None, optional
            Maximum contact distance.
        heavy_only : bool, optional
            Whether only heavy atoms should be analyzed.
        exclude_solvent : bool, optional
            Whether solvent atoms should be removed.
        exclude_ions : bool, optional
            Whether free receptor ions should be removed.
        exclude_same_residue : bool, optional
            Whether pairs assigned to the same residue should be ignored.
        block_size : int or None, optional
            Explicit processing block size.
        maximum_matrix_elements : int or None, optional
            Maximum full-matrix element count.
        sort_by_distance : bool, optional
            Whether contacts should be sorted by distance.
        maximum_contacts : int or None, optional
            Maximum number of returned contacts.
        metadata : mapping or None, optional
            Metadata copied to each contact.
        scene : bool, optional
            Whether ChimeraX scene coordinates should be preferred.
    
        Returns
        -------
        tuple of AtomContact
            Detected receptor-ligand contacts.
        
    """

    ligand_atoms, receptor_atoms = select_contact_collections(
        ligand,
        receptor,
        heavy_only=heavy_only,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        require_coordinate=False,
    )
    search_metadata = {} if metadata is None else dict(metadata)
    search_metadata.setdefault("collection_1_role", "ligand")
    search_metadata.setdefault("collection_2_role", "receptor")
    search_metadata.setdefault("heavy_only", bool(heavy_only))
    search_metadata.setdefault("exclude_solvent", bool(exclude_solvent))
    search_metadata.setdefault("exclude_ions", bool(exclude_ions))
    return find_atom_contacts(
        ligand_atoms,
        receptor_atoms,
        cutoff=cutoff,
        exclude_same_residue=exclude_same_residue,
        exclude_identical_atoms=True,
        block_size=block_size,
        maximum_matrix_elements=maximum_matrix_elements,
        sort_by_distance=sort_by_distance,
        maximum_contacts=maximum_contacts,
        classification=CONTACT_TYPE_CONTACT,
        metadata=search_metadata,
        scene=scene,
    )

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES = [
    "find_atom_contacts",
    "find_contacts",
    "closest_contact",
]

_register_public_names(_SECTION_6_PUBLIC_NAMES)


# =============================================================================
# End of Section 6
# =============================================================================


# =============================================================================
# Section 7 — Residue-level contact grouping
# =============================================================================


# -----------------------------------------------------------------------------
# Internal validation and residue-selection helpers
# -----------------------------------------------------------------------------

def _validate_contact_side(
    side: str,
) -> str:
    """Validate and normalize a contact side."""

    if not isinstance(
        side,
        str,
    ):
        raise TypeError(
            "side must be a string."
        )

    normalized_side = (
        side.strip().lower()
    )

    side_aliases = {
        "atom_1": "atom_1",
        "atom1": "atom_1",
        "first": "atom_1",
        "ligand": "atom_1",
        "lig": "atom_1",
        "atom_2": "atom_2",
        "atom2": "atom_2",
        "second": "atom_2",
        "receptor": "atom_2",
        "rec": "atom_2",
        "protein": "atom_2",
        "both": "both",
        "all": "both",
    }

    try:
        return side_aliases[
            normalized_side
        ]

    except KeyError as error:
        raise ValueError(
            "side must be one of: "
            "'atom_1', 'atom_2', 'ligand', "
            "'receptor' or 'both'."
        ) from error


def _validate_atom_contacts(
    contacts: Iterable[AtomContact],
    *,
    allow_empty: bool = True,
) -> Tuple[AtomContact, ...]:
    """Validate and normalize an atom-contact collection."""

    if contacts is None:
        raise TypeError(
            "contacts cannot be None."
        )

    if isinstance(
        contacts,
        (
            str,
            bytes,
            Mapping,
        ),
    ):
        raise TypeError(
            "contacts must be an iterable of "
            "AtomContact objects."
        )

    try:
        normalized_contacts = tuple(
            contacts
        )

    except TypeError as error:
        raise TypeError(
            "contacts must be iterable."
        ) from error

    for index, contact in enumerate(
        normalized_contacts
    ):
        if not isinstance(
            contact,
            AtomContact,
        ):
            raise TypeError(
                "contacts must contain only "
                "AtomContact instances. "
                f"Invalid item at index {index}."
            )

    if (
        not normalized_contacts
        and not allow_empty
    ):
        raise ValueError(
            "Contact collection cannot be empty."
        )

    return normalized_contacts


def _contact_residue_entries(
    contact: AtomContact,
    *,
    side: str,
    include_missing: bool = False,
) -> Tuple[
    Tuple[
        Optional[ResidueLike],
        AtomLike,
        str,
    ],
    ...,
]:
    """Return residue entries represented by a contact."""

    entries: List[
        Tuple[
            Optional[ResidueLike],
            AtomLike,
            str,
        ]
    ] = []

    if side in (
        "atom_1",
        "both",
    ):
        residue_1 = (
            contact.residue_1
            if contact.residue_1
            is not None
            else get_atom_residue(
                contact.atom_1
            )
        )

        if (
            residue_1 is not None
            or include_missing
        ):
            entries.append(
                (
                    residue_1,
                    contact.atom_1,
                    "atom_1",
                )
            )

    if side in (
        "atom_2",
        "both",
    ):
        residue_2 = (
            contact.residue_2
            if contact.residue_2
            is not None
            else get_atom_residue(
                contact.atom_2
            )
        )

        if (
            residue_2 is not None
            or include_missing
        ):
            entries.append(
                (
                    residue_2,
                    contact.atom_2,
                    "atom_2",
                )
            )

    return tuple(
        entries
    )


def _residue_group_key(
    residue: Optional[ResidueLike],
    atom: AtomLike,
    *,
    include_structure: bool,
) -> Tuple[Any, ...]:
    """Build an internal grouping key for a residue."""

    residue_key = (
        get_residue_contact_key(
            residue
        )
    )

    if not include_structure:
        return residue_key

    structure = get_atom_structure(
        atom
    )

    structure_identity = (
        None
        if structure is None
        else id(
            structure
        )
    )

    return (
        *residue_key,
        structure_identity,
    )


def _residue_contact_sort_key(
    result: ResidueContact,
) -> Tuple[
    str,
    int,
    str,
]:
    """Build a deterministic sorting key for residue-level results."""

    residue_name, residue_number, chain_id = (
        result.key
    )

    return (
        chain_id,
        residue_number,
        residue_name,
    )


# -----------------------------------------------------------------------------
# Contact grouping
# -----------------------------------------------------------------------------

def group_contacts_by_residue(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
    include_missing: bool = False,
    include_structure_identity: bool = True,
    sort_contacts: bool = True,
) -> Dict[
    ResidueContactKey,
    Tuple[
        AtomContact,
        ...,
    ],
]:
    """
    Group atom-level contacts by residue.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    side : str, optional
        Side used for grouping. ``"receptor"`` and ``"atom_2"`` group by
        the second atom; ``"ligand"`` and ``"atom_1"`` group by the first;
        ``"both"`` groups residues from both sides.
    include_missing : bool, optional
        Whether contacts without a residue should be grouped under the
        fallback key ``("UNK", -1, "")``.
    include_structure_identity : bool, optional
        Whether residues with identical name, number and chain but belonging
        to different structures should remain internally separated.
    sort_contacts : bool, optional
        Whether contacts inside each group should be ordered by distance.

    Returns
    -------
    dict
        Mapping from :class:`ResidueContactKey` to tuples of
        :class:`AtomContact`.

    Notes
    -----
    The returned public keys contain only residue name, number and chain.
    When structure identity is enabled, internally distinct residues with
    colliding public keys are merged because the public
    :class:`ResidueContactKey` does not contain a structure identifier.

    For ordinary ligand-receptor analyses involving one receptor structure,
    this does not create ambiguity.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    normalized_side = (
        _validate_contact_side(
            side
        )
    )

    internal_groups: DefaultDict[
        Tuple[Any, ...],
        List[AtomContact],
    ] = defaultdict(
        list
    )

    internal_public_keys: Dict[
        Tuple[Any, ...],
        ResidueContactKey,
    ] = {}

    for contact in normalized_contacts:
        entries = (
            _contact_residue_entries(
                contact,
                side=normalized_side,
                include_missing=include_missing,
            )
        )

        processed_keys: Set[
            Tuple[Any, ...]
        ] = set()

        for residue, atom, _ in entries:
            internal_key = (
                _residue_group_key(
                    residue,
                    atom,
                    include_structure=(
                        include_structure_identity
                    ),
                )
            )

            # Prevent a contact from being inserted twice when side="both"
            # and both atoms belong to the same residue.
            if internal_key in processed_keys:
                continue

            processed_keys.add(
                internal_key
            )

            public_key = (
                get_residue_contact_key(
                    residue
                )
            )

            internal_groups[
                internal_key
            ].append(
                contact
            )

            internal_public_keys[
                internal_key
            ] = public_key

    public_groups: DefaultDict[
        ResidueContactKey,
        List[AtomContact],
    ] = defaultdict(
        list
    )

    for (
        internal_key,
        grouped_contacts,
    ) in internal_groups.items():
        public_key = (
            internal_public_keys[
                internal_key
            ]
        )

        public_groups[
            public_key
        ].extend(
            grouped_contacts
        )

    result: Dict[
        ResidueContactKey,
        Tuple[
            AtomContact,
            ...,
        ],
    ] = {}

    for key in sorted(
        public_groups,
        key=lambda residue_key: (
            residue_key[
                2
            ],
            residue_key[
                1
            ],
            residue_key[
                0
            ],
        ),
    ):
        grouped_contacts = (
            public_groups[
                key
            ]
        )

        if sort_contacts:
            grouped_contacts.sort(
                key=_contact_sort_key
            )

        result[
            key
        ] = tuple(
            grouped_contacts
        )

    return result


# -----------------------------------------------------------------------------
# ResidueContact result construction
# -----------------------------------------------------------------------------

def residue_contacts(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
    include_missing: bool = False,
    include_structure_identity: bool = True,
    sort_contacts: bool = True,
    sort_residues: bool = True,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> Tuple[
    ResidueContact,
    ...,
]:
    """
    Convert atom contacts into residue-level contact results.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    side : str, optional
        Contact side used for residue grouping.
    include_missing : bool, optional
        Whether atoms without residues should be included.
    include_structure_identity : bool, optional
        Whether structure identity should be considered during grouping.
    sort_contacts : bool, optional
        Whether atom contacts should be sorted by distance within each
        residue.
    sort_residues : bool, optional
        Whether residue results should be sorted by chain, number and name.
    metadata : mapping or None, optional
        Metadata copied into each :class:`ResidueContact`.

    Returns
    -------
    tuple of ResidueContact
        Residue-level contact results.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    normalized_side = (
        _validate_contact_side(
            side
        )
    )

    grouped_contacts = (
        group_contacts_by_residue(
            normalized_contacts,
            side=normalized_side,
            include_missing=include_missing,
            include_structure_identity=(
                include_structure_identity
            ),
            sort_contacts=sort_contacts,
        )
    )

    residue_objects: Dict[
        ResidueContactKey,
        Optional[ResidueLike],
    ] = {}

    for contact in normalized_contacts:
        entries = (
            _contact_residue_entries(
                contact,
                side=normalized_side,
                include_missing=include_missing,
            )
        )

        for residue, _, _ in entries:
            key = get_residue_contact_key(
                residue
            )

            if key not in residue_objects:
                residue_objects[
                    key
                ] = residue

    base_metadata = (
        {}
        if metadata is None
        else dict(
            metadata
        )
    )

    base_metadata.setdefault(
        "grouped_side",
        normalized_side,
    )

    results: List[
        ResidueContact
    ] = []

    for (
        key,
        grouped_atom_contacts,
    ) in grouped_contacts.items():
        residue = residue_objects.get(
            key
        )

        residue_metadata = dict(
            base_metadata
        )

        residue_metadata[
            "contact_count"
        ] = len(
            grouped_atom_contacts
        )

        residue_metadata[
            "residue_name"
        ] = key[
            0
        ]

        residue_metadata[
            "residue_number"
        ] = key[
            1
        ]

        residue_metadata[
            "chain_id"
        ] = key[
            2
        ]

        results.append(
            ResidueContact(
                residue=residue,
                key=key,
                contacts=(
                    grouped_atom_contacts
                ),
                minimum_distance=None,
                metadata=(
                    residue_metadata
                ),
            )
        )

    if sort_residues:
        results.sort(
            key=_residue_contact_sort_key
        )

    return tuple(
        results
    )


# -----------------------------------------------------------------------------
# Contacting residue retrieval
# -----------------------------------------------------------------------------

def contacting_residues(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
    include_missing: bool = False,
    unique: bool = True,
    return_keys: bool = False,
    sort: bool = True,
) -> Union[
    Tuple[
        ResidueLike,
        ...,
    ],
    Tuple[
        ResidueContactKey,
        ...,
    ],
]:
    """
    Return residues involved in atom-level contacts.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    side : str, optional
        Side from which residues should be collected.
    include_missing : bool, optional
        Whether missing residues should be represented.
    unique : bool, optional
        Whether duplicate residues should be removed.
    return_keys : bool, optional
        Whether stable residue keys should be returned instead of raw
        residue objects.
    sort : bool, optional
        Whether results should be sorted by chain, number and name.

    Returns
    -------
    tuple
        Residue objects or :class:`ResidueContactKey` values.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    normalized_side = (
        _validate_contact_side(
            side
        )
    )

    entries: List[
        Tuple[
            ResidueContactKey,
            Optional[ResidueLike],
        ]
    ] = []

    seen_keys: Set[
        ResidueContactKey
    ] = set()

    for contact in normalized_contacts:
        residue_entries = (
            _contact_residue_entries(
                contact,
                side=normalized_side,
                include_missing=include_missing,
            )
        )

        for residue, _, _ in residue_entries:
            key = get_residue_contact_key(
                residue
            )

            if (
                unique
                and key in seen_keys
            ):
                continue

            seen_keys.add(
                key
            )

            entries.append(
                (
                    key,
                    residue,
                )
            )

    if sort:
        entries.sort(
            key=lambda entry: (
                entry[
                    0
                ][
                    2
                ],
                entry[
                    0
                ][
                    1
                ],
                entry[
                    0
                ][
                    0
                ],
            )
        )

    if return_keys:
        return tuple(
            key
            for key, _ in entries
        )

    return tuple(
        residue
        for _, residue in entries
        if (
            residue is not None
            or include_missing
        )
    )


# -----------------------------------------------------------------------------
# Residue-level summaries
# -----------------------------------------------------------------------------

def residue_contact_counts(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
    include_missing: bool = False,
    sort_by_count: bool = False,
    descending: bool = True,
) -> Dict[
    ResidueContactKey,
    int,
]:
    """
    Count atom-level contacts per residue.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    side : str, optional
        Contact side used for grouping.
    include_missing : bool, optional
        Whether missing residues should be included.
    sort_by_count : bool, optional
        Whether the result should be ordered by contact count instead of
        residue identity.
    descending : bool, optional
        Whether count sorting should use descending order.

    Returns
    -------
    dict
        Mapping from residue key to contact count.
    """

    groups = group_contacts_by_residue(
        contacts,
        side=side,
        include_missing=include_missing,
        sort_contacts=False,
    )

    items = [
        (
            key,
            len(
                grouped_contacts
            ),
        )
        for key, grouped_contacts
        in groups.items()
    ]

    if sort_by_count:
        items.sort(
            key=lambda item: (
                item[
                    1
                ],
                item[
                    0
                ][
                    2
                ],
                item[
                    0
                ][
                    1
                ],
                item[
                    0
                ][
                    0
                ],
            ),
            reverse=descending,
        )

    return dict(
        items
    )


def format_residue_contact_label(
    key: ResidueContactKey,
    *,
    include_chain: bool = True,
    chain_separator: str = ":",
) -> str:
    """
    Format a residue-contact key for reports.

    Parameters
    ----------
    key : ResidueContactKey
        Residue identifier.
    include_chain : bool, optional
        Whether a non-empty chain identifier should be included.
    chain_separator : str, optional
        Separator placed between chain and residue.

    Returns
    -------
    str
        Formatted residue label.

    Examples
    --------
    ``("TYR", 58, "A")`` becomes ``"A:TYR58"``.

    With ``include_chain=False``, it becomes ``"TYR58"``.
    """

    if (
        not isinstance(
            key,
            tuple,
        )
        or len(
            key
        )
        != 3
    ):
        raise TypeError(
            "key must be a three-item "
            "ResidueContactKey tuple."
        )

    residue_name = (
        _normalize_text_value(
            key[
                0
            ],
            default="UNK",
            uppercase=True,
        )
    )

    residue_number = key[
        1
    ]

    if isinstance(
        residue_number,
        bool,
    ) or not isinstance(
        residue_number,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "Residue number must be an integer."
        )

    chain_id = (
        _normalize_text_value(
            key[
                2
            ]
        )
    )

    residue_label = (
        f"{residue_name}"
        f"{int(residue_number)}"
    )

    if (
        include_chain
        and chain_id
    ):
        return (
            f"{chain_id}"
            f"{chain_separator}"
            f"{residue_label}"
        )

    return residue_label


def summarize_residue_contacts(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
    include_chain: bool = True,
    sort_by_count: bool = True,
    descending: bool = True,
) -> Tuple[
    Tuple[
        str,
        int,
    ],
    ...,
]:
    """
    Build compact residue-contact summaries.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    side : str, optional
        Contact side used for grouping.
    include_chain : bool, optional
        Whether chain identifiers should be included in labels.
    sort_by_count : bool, optional
        Whether residues should be sorted by contact count.
    descending : bool, optional
        Whether count sorting should be descending.

    Returns
    -------
    tuple
        Tuples containing ``(residue_label, contact_count)``.

    Examples
    --------
    A possible result is::

        (
            ("TYR58", 4),
            ("PHE77", 2),
            ("SER205", 1),
        )
    """

    counts = residue_contact_counts(
        contacts,
        side=side,
        sort_by_count=sort_by_count,
        descending=descending,
    )

    return tuple(
        (
            format_residue_contact_label(
                key,
                include_chain=(
                    include_chain
                ),
            ),
            count,
        )
        for key, count
        in counts.items()
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES = [
    "group_contacts_by_residue",
    "residue_contacts",
    "contacting_residues",
    "residue_contact_counts",
    "format_residue_contact_label",
    "summarize_residue_contacts",
]

_register_public_names(_SECTION_7_PUBLIC_NAMES)


# =============================================================================
# End of Section 7
# =============================================================================


# =============================================================================
# Section 8 — General contact classification
# =============================================================================


# -----------------------------------------------------------------------------
# van der Waals radii
# -----------------------------------------------------------------------------

_FALLBACK_VDW_RADII: Mapping[
    str,
    np.float64,
] = {
    "H": np.float64(1.20),
    "HE": np.float64(1.40),
    "LI": np.float64(1.82),
    "B": np.float64(1.92),
    "C": np.float64(1.70),
    "N": np.float64(1.55),
    "O": np.float64(1.52),
    "F": np.float64(1.47),
    "NE": np.float64(1.54),
    "NA": np.float64(2.27),
    "MG": np.float64(1.73),
    "AL": np.float64(1.84),
    "SI": np.float64(2.10),
    "P": np.float64(1.80),
    "S": np.float64(1.80),
    "CL": np.float64(1.75),
    "AR": np.float64(1.88),
    "K": np.float64(2.75),
    "CA": np.float64(2.31),
    "MN": np.float64(2.00),
    "FE": np.float64(2.00),
    "CO": np.float64(2.00),
    "NI": np.float64(1.97),
    "CU": np.float64(1.96),
    "ZN": np.float64(2.01),
    "BR": np.float64(1.85),
    "I": np.float64(1.98),
}


def _get_configured_vdw_radii(
) -> Mapping[
    str,
    np.float64,
]:
    """Return configured van der Waals radii."""

    candidate_names = (
        "VDW_RADII",
        "VAN_DER_WAALS_RADII",
        "ELEMENT_VDW_RADII",
        "DEFAULT_VDW_RADII",
    )

    configured_radii: Optional[
        Mapping[
            Any,
            Any,
        ]
    ] = None

    for attribute_name in candidate_names:
        value = getattr(
            config,
            attribute_name,
            None,
        )

        if value is not None:
            configured_radii = value
            break

    normalized_radii: Dict[
        str,
        np.float64,
    ] = dict(
        _FALLBACK_VDW_RADII
    )

    if configured_radii is None:
        return normalized_radii

    if not isinstance(
        configured_radii,
        Mapping,
    ):
        raise TypeError(
            "Configured van der Waals radii "
            "must be provided as a mapping."
        )

    for element, radius in (
        configured_radii.items()
    ):
        normalized_element = (
            _normalize_text_value(
                element,
                uppercase=True,
            )
        )

        if not normalized_element:
            continue

        if isinstance(
            radius,
            bool,
        ) or not isinstance(
            radius,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            continue

        radius_value = np.float64(
            radius
        )

        if (
            not np.isfinite(
                radius_value
            )
            or radius_value <= 0.0
        ):
            continue

        normalized_radii[
            normalized_element
        ] = radius_value

    return normalized_radii


def get_vdw_radius(
    atom_or_element: Any,
    *,
    default: Optional[Number] = None,
) -> Optional[np.float64]:
    """
    Return a van der Waals radius.

    Parameters
    ----------
    atom_or_element : Any
        Atom-like object or chemical element symbol.
    default : Number or None, optional
        Fallback radius when the element is unknown.

    Returns
    -------
    numpy.float64 or None
        van der Waals radius in angstroms.
    """

    if isinstance(
        atom_or_element,
        str,
    ):
        element = (
            _normalize_text_value(
                atom_or_element,
                uppercase=True,
            )
        )

    else:
        element = get_atom_element(
            atom_or_element,
            default="",
            infer_from_name=True,
        )

    radii = _get_configured_vdw_radii()

    radius = radii.get(
        element
    )

    if radius is not None:
        return np.float64(
            radius
        )

    if default is None:
        return None

    if isinstance(
        default,
        bool,
    ) or not isinstance(
        default,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        raise TypeError(
            "default radius must be numeric or None."
        )

    default_value = np.float64(
        default
    )

    if (
        not np.isfinite(
            default_value
        )
        or default_value <= 0.0
    ):
        raise ValueError(
            "default radius must be finite "
            "and greater than zero."
        )

    return default_value


def get_vdw_radius_sum(
    atom_1: AtomLike,
    atom_2: AtomLike,
    *,
    default_radius: Optional[Number] = None,
) -> Optional[np.float64]:
    """
    Return the sum of two atomic van der Waals radii.

    Parameters
    ----------
    atom_1 : AtomLike
        First atom.
    atom_2 : AtomLike
        Second atom.
    default_radius : Number or None, optional
        Radius used for unknown elements.

    Returns
    -------
    numpy.float64 or None
        Sum of radii, or ``None`` if at least one radius is unavailable.
    """

    radius_1 = get_vdw_radius(
        atom_1,
        default=default_radius,
    )

    radius_2 = get_vdw_radius(
        atom_2,
        default=default_radius,
    )

    if (
        radius_1 is None
        or radius_2 is None
    ):
        return None

    return np.float64(
        radius_1
        + radius_2
    )


# -----------------------------------------------------------------------------
# Classification tolerances
# -----------------------------------------------------------------------------

def _get_configured_float(
    candidate_names: Sequence[str],
    fallback: Number,
    *,
    minimum: Optional[Number] = None,
) -> np.float64:
    """Retrieve a numeric configuration value."""

    value: Any = fallback

    for attribute_name in candidate_names:
        configured_value = getattr(
            config,
            attribute_name,
            None,
        )

        if configured_value is not None:
            value = configured_value
            break

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        raise TypeError(
            "Configured contact-classification "
            "threshold must be numeric."
        )

    normalized_value = np.float64(
        value
    )

    if not np.isfinite(
        normalized_value
    ):
        raise ValueError(
            "Configured contact-classification "
            "threshold must be finite."
        )

    if (
        minimum is not None
        and normalized_value
        < np.float64(
            minimum
        )
    ):
        raise ValueError(
            "Configured contact-classification "
            f"threshold must be at least {minimum}."
        )

    return normalized_value


def get_clash_overlap_threshold(
) -> np.float64:
    """
    Return the minimum radius overlap classified as a steric clash.

    Returns
    -------
    numpy.float64
        Clash overlap threshold in angstroms.
    """

    return _get_configured_float(
        (
            "STERIC_CLASH_OVERLAP",
            "CLASH_OVERLAP_THRESHOLD",
            "DEFAULT_CLASH_OVERLAP",
        ),
        0.40,
        minimum=0.0,
    )


def get_close_contact_tolerance(
) -> np.float64:
    """
    Return the tolerance defining a close contact.

    Returns
    -------
    numpy.float64
        Distance tolerance in angstroms.
    """

    return _get_configured_float(
        (
            "CLOSE_CONTACT_TOLERANCE",
            "DEFAULT_CLOSE_CONTACT_TOLERANCE",
        ),
        0.20,
        minimum=0.0,
    )


def get_vdw_contact_tolerance(
) -> np.float64:
    """
    Return the tolerance defining a van der Waals contact.

    Returns
    -------
    numpy.float64
        Distance tolerance above the radius sum.
    """

    return _get_configured_float(
        (
            "VDW_CONTACT_TOLERANCE",
            "VAN_DER_WAALS_CONTACT_TOLERANCE",
            "DEFAULT_VDW_CONTACT_TOLERANCE",
        ),
        0.50,
        minimum=0.0,
    )


# -----------------------------------------------------------------------------
# General contact classification
# -----------------------------------------------------------------------------

def classify_contact(
    atom_1: AtomLike,
    atom_2: AtomLike,
    distance: Number,
    *,
    clash_overlap: Optional[Number] = None,
    close_tolerance: Optional[Number] = None,
    vdw_tolerance: Optional[Number] = None,
    default_radius: Optional[Number] = None,
) -> ContactClassification:
    """
    Classify a contact using only general geometric criteria.

    Parameters
    ----------
    atom_1 : AtomLike
        First atom.
    atom_2 : AtomLike
        Second atom.
    distance : Number
        Interatomic distance in angstroms.
    clash_overlap : Number or None, optional
        Minimum overlap below the van der Waals radius sum required for a
        steric clash.
    close_tolerance : Number or None, optional
        Tolerance around the van der Waals radius sum classified as a close
        contact.
    vdw_tolerance : Number or None, optional
        Maximum positive deviation from the radius sum classified as a
        van der Waals contact.
    default_radius : Number or None, optional
        Radius used for elements absent from the radius table.

    Returns
    -------
    ContactClassification
        One of ``steric_clash``, ``close_contact``,
        ``van_der_waals`` or ``unknown``.

    Notes
    -----
    This function intentionally does not classify hydrogen bonds,
    hydrophobic interactions, salt bridges or aromatic interactions.
    """

    if isinstance(
        distance,
        bool,
    ) or not isinstance(
        distance,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        raise TypeError(
            "distance must be numeric."
        )

    distance_value = np.float64(
        distance
    )

    if (
        not np.isfinite(
            distance_value
        )
        or distance_value < 0.0
    ):
        raise ValueError(
            "distance must be finite and non-negative."
        )

    radius_sum = get_vdw_radius_sum(
        atom_1,
        atom_2,
        default_radius=default_radius,
    )

    if radius_sum is None:
        return CONTACT_TYPE_UNKNOWN

    clash_threshold = (
        get_clash_overlap_threshold()
        if clash_overlap is None
        else _get_configured_float(
            (),
            clash_overlap,
            minimum=0.0,
        )
    )

    close_threshold = (
        get_close_contact_tolerance()
        if close_tolerance is None
        else _get_configured_float(
            (),
            close_tolerance,
            minimum=0.0,
        )
    )

    vdw_threshold = (
        get_vdw_contact_tolerance()
        if vdw_tolerance is None
        else _get_configured_float(
            (),
            vdw_tolerance,
            minimum=0.0,
        )
    )

    if close_threshold > vdw_threshold:
        raise ValueError(
            "close_tolerance cannot be greater "
            "than vdw_tolerance."
        )

    deviation = np.float64(
        distance_value
        - radius_sum
    )

    overlap = np.float64(
        radius_sum
        - distance_value
    )

    if overlap >= clash_threshold:
        return CONTACT_TYPE_CLASH

    if abs(
        deviation
    ) <= close_threshold:
        return CONTACT_TYPE_CLOSE_CONTACT

    if (
        deviation > close_threshold
        and deviation <= vdw_threshold
    ):
        return CONTACT_TYPE_VDW

    # A small overlap that does not reach the clash threshold is retained
    # as a close contact.
    if (
        deviation < -close_threshold
        and overlap < clash_threshold
    ):
        return CONTACT_TYPE_CLOSE_CONTACT

    return CONTACT_TYPE_UNKNOWN


def classify_atom_contact(
    contact: AtomContact,
    *,
    clash_overlap: Optional[Number] = None,
    close_tolerance: Optional[Number] = None,
    vdw_tolerance: Optional[Number] = None,
    default_radius: Optional[Number] = None,
    preserve_metadata: bool = True,
) -> AtomContact:
    """
    Return a classified copy of an atom contact.

    Parameters
    ----------
    contact : AtomContact
        Contact to classify.
    clash_overlap : Number or None, optional
        Steric-clash overlap threshold.
    close_tolerance : Number or None, optional
        Close-contact tolerance.
    vdw_tolerance : Number or None, optional
        van der Waals tolerance.
    default_radius : Number or None, optional
        Radius used for unknown elements.
    preserve_metadata : bool, optional
        Whether existing metadata should be retained.

    Returns
    -------
    AtomContact
        New immutable contact with updated classification.
    """

    if not isinstance(
        contact,
        AtomContact,
    ):
        raise TypeError(
            "contact must be an AtomContact instance."
        )

    radius_sum = get_vdw_radius_sum(
        contact.atom_1,
        contact.atom_2,
        default_radius=default_radius,
    )

    classification = classify_contact(
        contact.atom_1,
        contact.atom_2,
        contact.distance,
        clash_overlap=clash_overlap,
        close_tolerance=close_tolerance,
        vdw_tolerance=vdw_tolerance,
        default_radius=default_radius,
    )

    contact_metadata: Dict[
        str,
        Any,
    ] = (
        dict(
            contact.metadata
        )
        if preserve_metadata
        else {}
    )

    contact_metadata[
        "general_classification"
    ] = classification

    contact_metadata[
        "classification_method"
    ] = "van_der_waals_geometry"

    if radius_sum is not None:
        deviation = np.float64(
            contact.distance
            - radius_sum
        )

        contact_metadata[
            "vdw_radius_sum"
        ] = float(
            radius_sum
        )

        contact_metadata[
            "vdw_distance_deviation"
        ] = float(
            deviation
        )

        contact_metadata[
            "vdw_overlap"
        ] = float(
            max(
                np.float64(0.0),
                -deviation,
            )
        )

    else:
        contact_metadata[
            "vdw_radius_sum"
        ] = None

        contact_metadata[
            "vdw_distance_deviation"
        ] = None

        contact_metadata[
            "vdw_overlap"
        ] = None

    return replace(
        contact,
        classification=classification,
        metadata=contact_metadata,
    )


def classify_contacts(
    contacts: Iterable[AtomContact],
    *,
    clash_overlap: Optional[Number] = None,
    close_tolerance: Optional[Number] = None,
    vdw_tolerance: Optional[Number] = None,
    default_radius: Optional[Number] = None,
    sort_by_distance: bool = False,
) -> Tuple[
    AtomContact,
    ...,
]:
    """
    Classify an atom-contact collection.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Contacts to classify.
    clash_overlap : Number or None, optional
        Steric-clash overlap threshold.
    close_tolerance : Number or None, optional
        Close-contact tolerance.
    vdw_tolerance : Number or None, optional
        van der Waals contact tolerance.
    default_radius : Number or None, optional
        Radius used for unknown elements.
    sort_by_distance : bool, optional
        Whether results should be sorted by increasing distance.

    Returns
    -------
    tuple of AtomContact
        Classified immutable contacts.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    classified_contacts = [
        classify_atom_contact(
            contact,
            clash_overlap=clash_overlap,
            close_tolerance=close_tolerance,
            vdw_tolerance=vdw_tolerance,
            default_radius=default_radius,
        )
        for contact in normalized_contacts
    ]

    if sort_by_distance:
        classified_contacts.sort(
            key=_contact_sort_key
        )

    return tuple(
        classified_contacts
    )


# -----------------------------------------------------------------------------
# Classification filtering and summaries
# -----------------------------------------------------------------------------

def contacts_by_classification(
    contacts: Iterable[AtomContact],
    classification: ContactClassification,
) -> Tuple[
    AtomContact,
    ...,
]:
    """
    Select contacts with a general classification.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-contact collection.
    classification : ContactClassification
        Desired classification.

    Returns
    -------
    tuple of AtomContact
        Matching contacts.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    normalized_classification = (
        _normalize_text_value(
            classification,
            uppercase=False,
        ).lower()
    )

    if normalized_classification not in (
        _VALID_CONTACT_CLASSIFICATIONS
    ):
        raise ValueError(
            "Unknown contact classification: "
            f"{classification!r}."
        )

    return tuple(
        contact
        for contact in normalized_contacts
        if contact.classification
        == normalized_classification
    )


def contact_classification_counts(
    contacts: Iterable[AtomContact],
) -> Dict[
    ContactClassification,
    int,
]:
    """
    Count contacts by general classification.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-contact collection.

    Returns
    -------
    dict
        Classification counts.
    """

    normalized_contacts = (
        _validate_atom_contacts(
            contacts,
            allow_empty=True,
        )
    )

    counts: Dict[
        ContactClassification,
        int,
    ] = {
        CONTACT_TYPE_CLASH: 0,
        CONTACT_TYPE_CLOSE_CONTACT: 0,
        CONTACT_TYPE_VDW: 0,
        CONTACT_TYPE_UNKNOWN: 0,
    }

    for contact in normalized_contacts:
        classification = (
            contact.classification
        )

        if classification not in counts:
            classification = (
                CONTACT_TYPE_UNKNOWN
            )

        counts[
            classification
        ] += 1

    return counts


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES = [
    "get_vdw_radius",
    "get_vdw_radius_sum",
    "get_clash_overlap_threshold",
    "get_close_contact_tolerance",
    "get_vdw_contact_tolerance",
    "classify_contact",
    "classify_atom_contact",
    "classify_contacts",
    "contacts_by_classification",
    "contact_classification_counts",
]

_register_public_names(_SECTION_8_PUBLIC_NAMES)


# =============================================================================
# End of Section 8
# =============================================================================



# =============================================================================
# Section 9 — Contact summaries and statistics
# =============================================================================


# -----------------------------------------------------------------------------
# Distance extraction
# -----------------------------------------------------------------------------

def contact_distances(
    contacts: Iterable[AtomContact],
    *,
    finite_only: bool = True,
    sort: bool = False,
) -> FloatArray:
    """
    Extract distances from an atom-contact collection.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    finite_only : bool, optional
        Whether non-finite distance values should be excluded.
    sort : bool, optional
        Whether distances should be sorted in ascending order.

    Returns
    -------
    numpy.ndarray
        One-dimensional distance array with dtype ``numpy.float64``.
    """

    normalized_contacts = _validate_atom_contacts(
        contacts,
        allow_empty=True,
    )

    distances = np.asarray(
        [
            contact.distance
            for contact in normalized_contacts
        ],
        dtype=np.float64,
    )

    if finite_only:
        distances = distances[
            np.isfinite(
                distances
            )
        ]

    if sort:
        distances = np.sort(
            distances
        )

    return distances


# -----------------------------------------------------------------------------
# Generic numeric helpers
# -----------------------------------------------------------------------------



def _validate_percentiles(
    percentiles: Iterable[Number],
) -> Tuple[np.float64, ...]:
    """Validate percentile values."""

    if percentiles is None:
        raise TypeError(
            "percentiles cannot be None."
        )

    if isinstance(
        percentiles,
        (
            str,
            bytes,
            Mapping,
        ),
    ):
        raise TypeError(
            "percentiles must be an iterable of numbers."
        )

    try:
        percentile_values = tuple(
            percentiles
        )

    except TypeError as error:
        raise TypeError(
            "percentiles must be iterable."
        ) from error

    normalized_percentiles: List[np.float64] = []

    for index, percentile in enumerate(
        percentile_values
    ):
        if isinstance(
            percentile,
            bool,
        ) or not isinstance(
            percentile,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            raise TypeError(
                "Percentile values must be numeric. "
                f"Invalid item at index {index}."
            )

        percentile_value = np.float64(
            percentile
        )

        if (
            not np.isfinite(
                percentile_value
            )
            or percentile_value < 0.0
            or percentile_value > 100.0
        ):
            raise ValueError(
                "Percentile values must be finite "
                "and between 0 and 100."
            )

        normalized_percentiles.append(
            percentile_value
        )

    return tuple(
        normalized_percentiles
    )


def _percentile_label(
    percentile: Number,
) -> str:
    """Build a stable percentile dictionary key."""

    percentile_value = np.float64(
        percentile
    )

    if percentile_value.is_integer():
        return f"p{int(percentile_value)}"

    text = (
        f"{float(percentile_value):g}"
        .replace(
            ".",
            "_",
        )
        .replace(
            "-",
            "minus_",
        )
    )

    return f"p{text}"


def _distance_descriptive_statistics(
    distances: FloatArray,
    *,
    percentiles: Iterable[Number] = (
        25.0,
        50.0,
        75.0,
    ),
) -> Statistics:
    """Calculate descriptive statistics for a distance array."""

    distance_values = np.asarray(
        distances,
        dtype=np.float64,
    ).reshape(
        -1
    )

    finite_distances = distance_values[
        np.isfinite(
            distance_values
        )
    ]

    validated_percentiles = _validate_percentiles(
        percentiles
    )

    if finite_distances.size == 0:
        empty_statistics: Statistics = {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "range": None,
            "sum": 0.0,
        }

        for percentile in validated_percentiles:
            empty_statistics[
                _percentile_label(
                    percentile
                )
            ] = None

        return empty_statistics

    minimum = np.min(
        finite_distances
    )

    maximum = np.max(
        finite_distances
    )

    mean = np.mean(
        finite_distances,
        dtype=np.float64,
    )

    median = np.median(
        finite_distances
    )

    standard_deviation = np.std(
        finite_distances,
        ddof=0,
        dtype=np.float64,
    )

    variance = np.var(
        finite_distances,
        ddof=0,
        dtype=np.float64,
    )

    statistics: Statistics = {
        "count": int(
            finite_distances.size
        ),
        "minimum": float(
            minimum
        ),
        "maximum": float(
            maximum
        ),
        "mean": float(
            mean
        ),
        "median": float(
            median
        ),
        "standard_deviation": float(
            standard_deviation
        ),
        "variance": float(
            variance
        ),
        "range": float(
            maximum - minimum
        ),
        "sum": float(
            np.sum(
                finite_distances,
                dtype=np.float64,
            )
        ),
    }

    if validated_percentiles:
        percentile_values = np.percentile(
            finite_distances,
            validated_percentiles,
        )

        percentile_values = np.atleast_1d(
            percentile_values
        )

        for percentile, value in zip(
            validated_percentiles,
            percentile_values,
        ):
            statistics[
                _percentile_label(
                    percentile
                )
            ] = float(
                value
            )

    return statistics


# -----------------------------------------------------------------------------
# Distance distribution
# -----------------------------------------------------------------------------

def _validate_histogram_bins(
    bins: Union[
        int,
        Sequence[Number],
        str,
    ],
) -> Union[
    int,
    FloatArray,
    str,
]:
    """Validate a histogram-bin specification."""

    if isinstance(
        bins,
        bool,
    ):
        raise TypeError(
            "bins cannot be a boolean value."
        )

    if isinstance(
        bins,
        (
            int,
            np.integer,
        ),
    ):
        normalized_bins = int(
            bins
        )

        if normalized_bins <= 0:
            raise ValueError(
                "Integer bins must be greater than zero."
            )

        return normalized_bins

    if isinstance(
        bins,
        str,
    ):
        normalized_method = (
            bins.strip().lower()
        )

        valid_methods = {
            "auto",
            "fd",
            "doane",
            "scott",
            "stone",
            "rice",
            "sturges",
            "sqrt",
        }

        if normalized_method not in valid_methods:
            raise ValueError(
                "Unsupported histogram-bin method: "
                f"{bins!r}."
            )

        return normalized_method

    try:
        bin_edges = np.asarray(
            tuple(
                bins
            ),
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "bins must be an integer, a supported "
            "method string or a sequence of bin edges."
        ) from error

    if (
        bin_edges.ndim != 1
        or bin_edges.size < 2
    ):
        raise ValueError(
            "Explicit histogram bins must contain "
            "at least two edges."
        )

    if not np.all(
        np.isfinite(
            bin_edges
        )
    ):
        raise ValueError(
            "Histogram bin edges must be finite."
        )

    if not np.all(
        np.diff(
            bin_edges
        )
        > 0.0
    ):
        raise ValueError(
            "Histogram bin edges must be strictly increasing."
        )

    return bin_edges


def contact_distance_distribution(
    contacts: Iterable[AtomContact],
    *,
    bins: Union[
        int,
        Sequence[Number],
        str,
    ] = "auto",
    distance_range: Optional[
        Tuple[
            Number,
            Number,
        ]
    ] = None,
    density: bool = False,
    include_values: bool = False,
) -> Statistics:
    """
    Calculate the distance distribution of atom contacts.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    bins : int, sequence of Number or str, optional
        Histogram-bin specification accepted by ``numpy.histogram``.
    distance_range : tuple of Number or None, optional
        Lower and upper histogram limits. When omitted, the observed range
        is used.
    density : bool, optional
        Whether histogram values should represent probability density.
    include_values : bool, optional
        Whether the raw finite distances should be included in the result.

    Returns
    -------
    Statistics
        Serializable histogram and distribution metadata.

    Notes
    -----
    ``density=False`` returns contact counts per bin. ``density=True`` returns
    probability-density values and still includes the independent
    ``sample_count`` field.
    """

    normalized_bins = _validate_histogram_bins(
        bins
    )

    distances = contact_distances(
        contacts,
        finite_only=True,
        sort=False,
    )

    normalized_range: Optional[
        Tuple[
            np.float64,
            np.float64,
        ]
    ] = None

    if distance_range is not None:
        if (
            not isinstance(
                distance_range,
                (
                    tuple,
                    list,
                ),
            )
            or len(
                distance_range
            )
            != 2
        ):
            raise TypeError(
                "distance_range must be a two-item tuple "
                "or list."
            )

        lower = _validate_contact_cutoff(
            distance_range[
                0
            ],
            name="distance_range lower bound",
            allow_zero=True,
        )

        upper = _validate_contact_cutoff(
            distance_range[
                1
            ],
            name="distance_range upper bound",
            allow_zero=True,
        )

        if upper <= lower:
            raise ValueError(
                "The upper distance-range bound must "
                "be greater than the lower bound."
            )

        normalized_range = (
            lower,
            upper,
        )

    if distances.size == 0:
        if isinstance(
            normalized_bins,
            np.ndarray,
        ):
            bin_edges = normalized_bins

        elif (
            normalized_range is not None
            and isinstance(
                normalized_bins,
                int,
            )
        ):
            bin_edges = np.linspace(
                normalized_range[
                    0
                ],
                normalized_range[
                    1
                ],
                normalized_bins + 1,
                dtype=np.float64,
            )

        else:
            bin_edges = np.asarray(
                [],
                dtype=np.float64,
            )

        empty_result: Statistics = {
            "sample_count": 0,
            "density": bool(
                density
            ),
            "bin_count": max(
                int(
                    bin_edges.size
                )
                - 1,
                0,
            ),
            "bin_edges": [
                float(
                    value
                )
                for value in bin_edges
            ],
            "bin_centers": [],
            "bin_widths": [],
            "counts": [],
            "frequencies": [],
            "cumulative_counts": [],
            "cumulative_frequencies": [],
            "minimum": None,
            "maximum": None,
        }

        if include_values:
            empty_result[
                "values"
            ] = []

        return empty_result

    histogram_values, bin_edges = np.histogram(
        distances,
        bins=normalized_bins,
        range=normalized_range,
        density=density,
    )

    raw_counts, _ = np.histogram(
        distances,
        bins=bin_edges,
        density=False,
    )

    sample_count = int(
        distances.size
    )

    frequencies = (
        raw_counts.astype(
            np.float64
        )
        / np.float64(
            sample_count
        )
    )

    cumulative_counts = np.cumsum(
        raw_counts,
        dtype=np.int64,
    )

    cumulative_frequencies = np.cumsum(
        frequencies,
        dtype=np.float64,
    )

    bin_widths = np.diff(
        bin_edges
    )

    bin_centers = (
        bin_edges[
            :-1
        ]
        + bin_edges[
            1:
        ]
    ) / np.float64(
        2.0
    )

    result: Statistics = {
        "sample_count": sample_count,
        "density": bool(
            density
        ),
        "bin_count": int(
            raw_counts.size
        ),
        "bin_edges": [
            float(
                value
            )
            for value in bin_edges
        ],
        "bin_centers": [
            float(
                value
            )
            for value in bin_centers
        ],
        "bin_widths": [
            float(
                value
            )
            for value in bin_widths
        ],
        "counts": [
            int(
                value
            )
            for value in raw_counts
        ],
        "histogram_values": [
            float(
                value
            )
            for value in histogram_values
        ],
        "frequencies": [
            float(
                value
            )
            for value in frequencies
        ],
        "cumulative_counts": [
            int(
                value
            )
            for value in cumulative_counts
        ],
        "cumulative_frequencies": [
            float(
                value
            )
            for value in cumulative_frequencies
        ],
        "minimum": float(
            np.min(
                distances
            )
        ),
        "maximum": float(
            np.max(
                distances
            )
        ),
    }

    if include_values:
        result[
            "values"
        ] = [
            float(
                value
            )
            for value in distances
        ]

    return result


# -----------------------------------------------------------------------------
# Residue statistics
# -----------------------------------------------------------------------------

def _residue_statistics(
    contacts: Iterable[AtomContact],
    *,
    side: str = "receptor",
) -> Statistics:
    """Calculate residue-level contact statistics."""

    grouped_results = residue_contacts(
        contacts,
        side=side,
        include_missing=False,
        sort_contacts=True,
        sort_residues=True,
    )

    contact_counts = np.asarray(
        [
            result.contact_count
            for result in grouped_results
        ],
        dtype=np.float64,
    )

    if contact_counts.size == 0:
        return {
            "residue_count": 0,
            "mean_contacts_per_residue": None,
            "median_contacts_per_residue": None,
            "maximum_contacts_per_residue": None,
            "minimum_contacts_per_residue": None,
            "most_contacted_residue": None,
            "most_contacted_residue_count": 0,
            "closest_residue": None,
            "closest_residue_distance": None,
        }

    most_contacted_result = min(
        grouped_results,
        key=lambda result: (
            -result.contact_count,
            result.minimum_distance,
            _residue_contact_sort_key(
                result
            ),
        ),
    )

    closest_result = min(
        grouped_results,
        key=lambda result: (
            result.minimum_distance,
            _residue_contact_sort_key(
                result
            ),
        ),
    )

    return {
        "residue_count": len(
            grouped_results
        ),
        "mean_contacts_per_residue": float(
            np.mean(
                contact_counts,
                dtype=np.float64,
            )
        ),
        "median_contacts_per_residue": float(
            np.median(
                contact_counts
            )
        ),
        "maximum_contacts_per_residue": int(
            np.max(
                contact_counts
            )
        ),
        "minimum_contacts_per_residue": int(
            np.min(
                contact_counts
            )
        ),
        "most_contacted_residue": (
            format_residue_contact_label(
                most_contacted_result.key,
                include_chain=True,
            )
        ),
        "most_contacted_residue_count": (
            most_contacted_result.contact_count
        ),
        "closest_residue": (
            format_residue_contact_label(
                closest_result.key,
                include_chain=True,
            )
        ),
        "closest_residue_distance": float(
            closest_result.minimum_distance
        ),
    }


# -----------------------------------------------------------------------------
# Complete contact statistics
# -----------------------------------------------------------------------------

def contact_statistics(
    contacts: Iterable[AtomContact],
    *,
    residue_side: str = "receptor",
    percentiles: Iterable[Number] = (
        25.0,
        50.0,
        75.0,
    ),
    include_residue_statistics: bool = True,
    include_classification_statistics: bool = True,
) -> Statistics:
    """
    Calculate descriptive statistics for atom contacts.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    residue_side : str, optional
        Side used for residue-level statistics.
    percentiles : iterable of Number, optional
        Distance percentiles expressed from 0 to 100.
    include_residue_statistics : bool, optional
        Whether residue-level statistics should be included.
    include_classification_statistics : bool, optional
        Whether general classification counts should be included.

    Returns
    -------
    Statistics
        Serializable contact statistics.

    Notes
    -----
    Standard deviation and variance are calculated as population statistics
    using ``ddof=0`` because the contacts generally represent the complete
    set detected for one docking pose, not a statistical sample.
    """

    normalized_contacts = _validate_atom_contacts(
        contacts,
        allow_empty=True,
    )

    normalized_side = _validate_contact_side(
        residue_side
    )

    distances = contact_distances(
        normalized_contacts,
        finite_only=True,
        sort=False,
    )

    distance_statistics = (
        _distance_descriptive_statistics(
            distances,
            percentiles=percentiles,
        )
    )

    contacting_atom_1_ids = {
        id(
            contact.atom_1
        )
        for contact in normalized_contacts
    }

    contacting_atom_2_ids = {
        id(
            contact.atom_2
        )
        for contact in normalized_contacts
    }

    within_cutoff_count = sum(
        1
        for contact in normalized_contacts
        if contact.is_contact
    )

    outside_cutoff_count = (
        len(
            normalized_contacts
        )
        - within_cutoff_count
    )

    statistics: Statistics = {
        "contact_count": len(
            normalized_contacts
        ),
        "has_contacts": bool(
            normalized_contacts
        ),
        "within_cutoff_count": (
            within_cutoff_count
        ),
        "outside_cutoff_count": (
            outside_cutoff_count
        ),
        "unique_atom_1_count": len(
            contacting_atom_1_ids
        ),
        "unique_atom_2_count": len(
            contacting_atom_2_ids
        ),
        "distance": distance_statistics,
    }

    if normalized_contacts:
        closest = min(
            normalized_contacts,
            key=_contact_sort_key,
        )

        farthest = max(
            normalized_contacts,
            key=_contact_sort_key,
        )

        statistics[
            "closest_contact"
        ] = {
            "atom_1": get_atom_identifier(
                closest.atom_1
            ),
            "atom_2": get_atom_identifier(
                closest.atom_2
            ),
            "distance": float(
                closest.distance
            ),
            "classification": (
                closest.classification
            ),
        }

        statistics[
            "farthest_contact"
        ] = {
            "atom_1": get_atom_identifier(
                farthest.atom_1
            ),
            "atom_2": get_atom_identifier(
                farthest.atom_2
            ),
            "distance": float(
                farthest.distance
            ),
            "classification": (
                farthest.classification
            ),
        }

    else:
        statistics[
            "closest_contact"
        ] = None

        statistics[
            "farthest_contact"
        ] = None

    if include_classification_statistics:
        classification_counts = (
            contact_classification_counts(
                normalized_contacts
            )
        )

        statistics[
            "classification_counts"
        ] = dict(
            classification_counts
        )

        total_contacts = len(
            normalized_contacts
        )

        statistics[
            "classification_frequencies"
        ] = {
            classification: (
                float(
                    count
                    / total_contacts
                )
                if total_contacts
                else 0.0
            )
            for classification, count
            in classification_counts.items()
        }

        statistics[
            "clash_count"
        ] = classification_counts.get(
            CONTACT_TYPE_CLASH,
            0,
        )

        statistics[
            "has_clashes"
        ] = bool(
            statistics[
                "clash_count"
            ]
        )

    if include_residue_statistics:
        statistics[
            "residues"
        ] = _residue_statistics(
            normalized_contacts,
            side=normalized_side,
        )

    return statistics


# -----------------------------------------------------------------------------
# Compact summaries
# -----------------------------------------------------------------------------

def summarize_contacts(
    contacts: Iterable[AtomContact],
    *,
    residue_side: str = "receptor",
    include_chain: bool = True,
    maximum_residues: Optional[int] = None,
    sort_residues_by_count: bool = True,
    include_distances: bool = True,
    include_classifications: bool = True,
    include_residues: bool = True,
) -> Statistics:
    """
    Build a compact summary of atom contacts.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    residue_side : str, optional
        Side used for residue summaries.
    include_chain : bool, optional
        Whether residue labels should contain chain identifiers.
    maximum_residues : int or None, optional
        Maximum number of residue summaries returned.
    sort_residues_by_count : bool, optional
        Whether residues should be ordered by decreasing contact count.
    include_distances : bool, optional
        Whether compact distance statistics should be included.
    include_classifications : bool, optional
        Whether classification counts should be included.
    include_residues : bool, optional
        Whether residue-level summaries should be included.

    Returns
    -------
    Statistics
        Compact serializable summary.
    """

    normalized_contacts = _validate_atom_contacts(
        contacts,
        allow_empty=True,
    )

    normalized_side = _validate_contact_side(
        residue_side
    )

    if maximum_residues is not None:
        if isinstance(
            maximum_residues,
            bool,
        ) or not isinstance(
            maximum_residues,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "maximum_residues must be an integer or None."
            )

        maximum_residues = int(
            maximum_residues
        )

        if maximum_residues < 0:
            raise ValueError(
                "maximum_residues cannot be negative."
            )

    summary: Statistics = {
        "contact_count": len(
            normalized_contacts
        ),
        "has_contacts": bool(
            normalized_contacts
        ),
    }

    if include_distances:
        distances = contact_distances(
            normalized_contacts,
            finite_only=True,
            sort=False,
        )

        descriptive = (
            _distance_descriptive_statistics(
                distances,
                percentiles=(
                    25.0,
                    50.0,
                    75.0,
                ),
            )
        )

        summary[
            "distance"
        ] = {
            "minimum": descriptive[
                "minimum"
            ],
            "maximum": descriptive[
                "maximum"
            ],
            "mean": descriptive[
                "mean"
            ],
            "median": descriptive[
                "median"
            ],
            "standard_deviation": descriptive[
                "standard_deviation"
            ],
            "p25": descriptive[
                "p25"
            ],
            "p75": descriptive[
                "p75"
            ],
        }

    if include_classifications:
        summary[
            "classification_counts"
        ] = contact_classification_counts(
            normalized_contacts
        )

    if include_residues:
        residue_summary = list(
            summarize_residue_contacts(
                normalized_contacts,
                side=normalized_side,
                include_chain=include_chain,
                sort_by_count=(
                    sort_residues_by_count
                ),
                descending=True,
            )
        )

        total_residue_count = len(
            residue_summary
        )

        if maximum_residues is not None:
            residue_summary = residue_summary[
                :maximum_residues
            ]

        summary[
            "contacting_residue_count"
        ] = total_residue_count

        summary[
            "residue_contacts"
        ] = [
            {
                "residue": label,
                "contact_count": count,
            }
            for label, count
            in residue_summary
        ]

        summary[
            "residue_summary_truncated"
        ] = bool(
            maximum_residues is not None
            and total_residue_count
            > maximum_residues
        )

    return summary


def format_contact_summary(
    contacts: Iterable[AtomContact],
    *,
    residue_side: str = "receptor",
    include_chain: bool = False,
    maximum_residues: Optional[int] = None,
    distance_precision: int = 2,
) -> str:
    """
    Format contacts as a human-readable multiline summary.

    Parameters
    ----------
    contacts : iterable of AtomContact
        Atom-level contacts.
    residue_side : str, optional
        Side used for residue grouping.
    include_chain : bool, optional
        Whether chain identifiers should be shown.
    maximum_residues : int or None, optional
        Maximum number of residues displayed.
    distance_precision : int, optional
        Number of decimal places used for distance values.

    Returns
    -------
    str
        Human-readable contact summary.
    """

    if isinstance(
        distance_precision,
        bool,
    ) or not isinstance(
        distance_precision,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "distance_precision must be an integer."
        )

    distance_precision = int(
        distance_precision
    )

    if distance_precision < 0:
        raise ValueError(
            "distance_precision cannot be negative."
        )

    normalized_contacts = _validate_atom_contacts(
        contacts,
        allow_empty=True,
    )

    summary = summarize_contacts(
        normalized_contacts,
        residue_side=residue_side,
        include_chain=include_chain,
        maximum_residues=maximum_residues,
        include_distances=True,
        include_classifications=True,
        include_residues=True,
    )

    contact_count = int(
        summary[
            "contact_count"
        ]
    )

    lines: List[str] = [
        (
            f"Contacts: {contact_count}"
        ),
        (
            "Contacting residues: "
            f"{summary['contacting_residue_count']}"
        ),
    ]

    distance_summary = summary.get(
        "distance",
        {}
    )

    minimum_distance = distance_summary.get(
        "minimum"
    )

    mean_distance = distance_summary.get(
        "mean"
    )

    median_distance = distance_summary.get(
        "median"
    )

    if minimum_distance is not None:
        lines.append(
            "Minimum distance: "
            f"{minimum_distance:.{distance_precision}f} Å"
        )

    if mean_distance is not None:
        lines.append(
            "Mean distance: "
            f"{mean_distance:.{distance_precision}f} Å"
        )

    if median_distance is not None:
        lines.append(
            "Median distance: "
            f"{median_distance:.{distance_precision}f} Å"
        )

    classification_counts = summary.get(
        "classification_counts",
        {},
    )

    if classification_counts:
        lines.append(
            "Classifications: "
            + ", ".join(
                (
                    f"{classification}="
                    f"{count}"
                )
                for classification, count
                in classification_counts.items()
            )
        )

    residue_summaries = summary.get(
        "residue_contacts",
        [],
    )

    if residue_summaries:
        lines.append(
            "Residues:"
        )

        for residue_summary in residue_summaries:
            count = int(
                residue_summary[
                    "contact_count"
                ]
            )

            plural_suffix = (
                ""
                if count == 1
                else "s"
            )

            lines.append(
                "  "
                f"{residue_summary['residue']} — "
                f"{count} contact"
                f"{plural_suffix}"
            )

    if summary.get(
        "residue_summary_truncated",
        False,
    ):
        lines.append(
            "  ..."
        )

    return "\n".join(
        lines
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES = [
    "contact_distances",
    "contact_distance_distribution",
    "contact_statistics",
    "summarize_contacts",
    "format_contact_summary",
]

_register_public_names(_SECTION_9_PUBLIC_NAMES)


# =============================================================================
# End of Section 9
# =============================================================================


# =============================================================================
# Section 10 — DockModel integration
# =============================================================================


# -----------------------------------------------------------------------------
# DockModel component retrieval
# -----------------------------------------------------------------------------

def _get_dock_model_value(
    dock_model: DockModel,
    candidate_names: Sequence[str],
    *,
    default: Any = None,
    call_if_callable: bool = False,
) -> Any:
    """Retrieve a value from a DockModel-like object."""

    if dock_model is None:
        return default

    for name in candidate_names:
        value: Any = None
        found = False

        if isinstance(
            dock_model,
            Mapping,
        ):
            if name in dock_model:
                value = dock_model[
                    name
                ]
                found = True

        elif hasattr(
            dock_model,
            name,
        ):
            try:
                value = getattr(
                    dock_model,
                    name,
                )
                found = True

            except Exception:
                continue

        if not found:
            continue

        if (
            call_if_callable
            and callable(
                value
            )
        ):
            try:
                value = value()

            except TypeError:
                continue

        if value is not None:
            return value

    return default


def get_dock_model_receptor(
    dock_model: DockModel,
    *,
    default: Any = None,
) -> Any:
    """
    Retrieve the receptor associated with a dock model.

    Parameters
    ----------
    dock_model : DockModel
        Dock model.
    default : Any, optional
        Fallback value.

    Returns
    -------
    Any
        Receptor-like object.

    Notes
    -----
    Several aliases are accepted to keep the contact module independent of
    small changes in the concrete ``DockModel`` implementation.
    """

    return _get_dock_model_value(
        dock_model,
        (
            "receptor",
            "receptor_model",
            "protein",
            "protein_model",
            "target",
            "target_model",
            "macromolecule",
        ),
        default=default,
        call_if_callable=True,
    )


def get_dock_model_ligand(
    dock_model: DockModel,
    *,
    default: Any = None,
) -> Any:
    """Retrieve the ligand source associated with a dock model."""

    return _get_dock_model_value(
        dock_model,
        (
            "ligand",
            "ligand_model",
            "ligand_residue",
            "ligand_atoms",
        ),
        default=default,
        call_if_callable=True,
    )


def get_dock_model_pose(
    dock_model: DockModel,
    *,
    pose_index: Optional[int] = None,
    default: Any = None,
) -> Any:
    """
    Retrieve a pose associated with a dock model.

    Parameters
    ----------
    dock_model : DockModel
        Dock model.
    pose_index : int or None, optional
        Pose index when the model contains multiple poses.
    default : Any, optional
        Fallback value.

    Returns
    -------
    Any
        Pose-like object.

    Raises
    ------
    TypeError
        If ``pose_index`` is not an integer or ``None``.
    IndexError
        If the requested pose index is unavailable.
    """

    if (
        pose_index is not None
        and (
            isinstance(
                pose_index,
                bool,
            )
            or not isinstance(
                pose_index,
                (
                    int,
                    np.integer,
                ),
            )
        )
    ):
        raise TypeError(
            "pose_index must be an integer or None."
        )

    normalized_pose_index = (
        None
        if pose_index is None
        else int(
            pose_index
        )
    )

    direct_pose = _get_dock_model_value(
        dock_model,
        (
            "pose",
            "active_pose",
            "current_pose",
            "selected_pose",
            "ligand",
            "ligand_model",
        ),
        default=None,
        call_if_callable=True,
    )

    if (
        direct_pose is not None
        and normalized_pose_index is None
    ):
        return direct_pose

    poses = _get_dock_model_value(
        dock_model,
        (
            "poses",
            "pose_models",
            "ligand_poses",
            "models",
            "conformations",
        ),
        default=None,
        call_if_callable=True,
    )

    if poses is None:
        if direct_pose is not None:
            if normalized_pose_index in (
                None,
                0,
            ):
                return direct_pose

            raise IndexError(
                "The dock model contains only one directly "
                "accessible pose."
            )

        return default

    if isinstance(
        poses,
        Mapping,
    ):
        if normalized_pose_index is None:
            try:
                return next(
                    iter(
                        poses.values()
                    )
                )

            except StopIteration:
                return default

        possible_keys = (
            normalized_pose_index,
            str(
                normalized_pose_index
            ),
            normalized_pose_index + 1,
            str(
                normalized_pose_index + 1
            ),
        )

        for key in possible_keys:
            if key in poses:
                return poses[
                    key
                ]

        raise IndexError(
            f"Pose index {normalized_pose_index} "
            "was not found."
        )

    try:
        normalized_poses = tuple(
            poses
        )

    except TypeError:
        if normalized_pose_index in (
            None,
            0,
        ):
            return poses

        raise IndexError(
            f"Pose index {normalized_pose_index} "
            "is unavailable."
        )

    if not normalized_poses:
        return default

    selected_index = (
        0
        if normalized_pose_index is None
        else normalized_pose_index
    )

    try:
        return normalized_poses[
            selected_index
        ]

    except IndexError as error:
        raise IndexError(
            f"Pose index {selected_index} is outside "
            f"the available range of {len(normalized_poses)} poses."
        ) from error


def get_dock_model_identifier(
    dock_model: DockModel,
    *,
    default: str = "dock_model",
) -> str:
    """
    Retrieve a stable human-readable dock-model identifier.

    Parameters
    ----------
    dock_model : DockModel
        Dock model.
    default : str, optional
        Fallback identifier.

    Returns
    -------
    str
        Dock-model identifier.
    """

    identifier = _get_dock_model_value(
        dock_model,
        (
            "identifier",
            "model_id",
            "id",
            "name",
            "label",
            "title",
        ),
        default=default,
        call_if_callable=False,
    )

    normalized_identifier = _normalize_text_value(
        identifier,
        default=default,
    )

    return (
        normalized_identifier
        if normalized_identifier
        else default
    )


def get_pose_identifier(
    pose: Any,
    *,
    pose_index: Optional[int] = None,
    default: str = "pose",
) -> str:
    """
    Retrieve a stable pose identifier.

    Parameters
    ----------
    pose : Any
        Pose-like object.
    pose_index : int or None, optional
        Pose index used as a fallback.
    default : str, optional
        Base fallback identifier.

    Returns
    -------
    str
        Pose identifier.
    """

    identifier = _get_object_value(
        pose,
        (
            "identifier",
            "pose_id",
            "model_id",
            "id",
            "name",
            "label",
            "title",
        ),
        default=None,
        call_if_callable=False,
    )

    normalized_identifier = _normalize_text_value(
        identifier,
        default="",
    )

    if normalized_identifier:
        return normalized_identifier

    if pose_index is not None:
        return (
            f"{default}_{int(pose_index)}"
        )

    return default


# -----------------------------------------------------------------------------
# Analysis attachment helpers
# -----------------------------------------------------------------------------

def _set_object_value(
    target: Any,
    name: str,
    value: Any,
) -> bool:
    """Attempt to assign a value to an object or mutable mapping."""

    if target is None:
        return False

    if isinstance(
        target,
        MutableMapping,
    ):
        try:
            target[
                name
            ] = value

            return True

        except Exception:
            return False

    try:
        setattr(
            target,
            name,
            value,
        )

        return True

    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return False


def _get_or_create_analysis_mapping(
    target: Any,
    *,
    attribute_name: str,
) -> Optional[
    MutableMapping[
        str,
        Any,
    ]
]:
    """Retrieve or create an analysis-result mapping."""

    existing_value = _get_object_value(
        target,
        (
            attribute_name,
        ),
        default=None,
        call_if_callable=False,
    )

    if isinstance(
        existing_value,
        MutableMapping,
    ):
        return existing_value

    new_mapping: Dict[
        str,
        Any,
    ] = {}

    if _set_object_value(
        target,
        attribute_name,
        new_mapping,
    ):
        return new_mapping

    return None


def attach_contact_analysis(
    dock_model: DockModel,
    result: ContactAnalysisResult,
    *,
    pose: Any = None,
    pose_identifier: Optional[str] = None,
    attribute_name: str = "contact_analyses",
    attach_to_pose: bool = True,
    overwrite: bool = True,
) -> bool:
    """Attach contact results to DockModel-compatible storage."""

    if not isinstance(
        result,
        ContactAnalysisResult,
    ):
        raise TypeError(
            "result must be a ContactAnalysisResult instance."
        )

    normalized_attribute_name = _normalize_text_value(
        attribute_name,
        default="contact_analyses",
    )

    if not normalized_attribute_name:
        raise ValueError(
            "attribute_name cannot be empty."
        )

    normalized_pose_identifier = _normalize_text_value(
        pose_identifier,
        default="pose",
    )

    attached = False

    analyses = _get_or_create_analysis_mapping(
        dock_model,
        attribute_name=normalized_attribute_name,
    )

    if analyses is not None and (
        overwrite
        or normalized_pose_identifier not in analyses
    ):
        analyses[
            normalized_pose_identifier
        ] = result
        attached = True

    existing_contacts = _get_object_value(
        dock_model,
        (
            "contacts",
        ),
        default=None,
        call_if_callable=False,
    )

    try:
        existing_contact_count = (
            0
            if existing_contacts is None
            else len(
                existing_contacts
            )
        )
    except TypeError:
        existing_contact_count = 1

    dock_contacts_attached = False

    if overwrite or existing_contact_count == 0:
        dock_contacts_attached = _set_object_value(
            dock_model,
            "contacts",
            list(
                result.contacts
            ),
        )
        attached = (
            attached
            or dock_contacts_attached
        )

    if dock_contacts_attached:
        update_statistics = getattr(
            dock_model,
            "update_statistics",
            None,
        )

        if callable(
            update_statistics
        ):
            try:
                update_statistics(
                    dict(
                        result.statistics
                    )
                )
            except TypeError:
                update_statistics()
        else:
            statistics = _get_object_value(
                dock_model,
                (
                    "statistics",
                ),
                default=None,
                call_if_callable=False,
            )

            if isinstance(
                statistics,
                MutableMapping,
            ):
                statistics.update(
                    dict(
                        result.statistics
                    )
                )
                statistics[
                    "contacts"
                ] = result.contact_count

    if attach_to_pose and pose is not None:
        existing_pose_result = _get_object_value(
            pose,
            (
                "contact_analysis",
            ),
            default=None,
            call_if_callable=False,
        )

        if overwrite or existing_pose_result is None:
            attached = (
                _set_object_value(
                    pose,
                    "contact_analysis",
                    result,
                )
                or attached
            )

        existing_pose_contacts = _get_object_value(
            pose,
            (
                "contacts",
            ),
            default=None,
            call_if_callable=False,
        )

        if overwrite or existing_pose_contacts is None:
            attached = (
                _set_object_value(
                    pose,
                    "contacts",
                    result.contacts,
                )
                or attached
            )

    return attached



def get_attached_contact_analysis(
    dock_model: DockModel,
    *,
    pose_identifier: Optional[str] = None,
    attribute_name: str = "contact_analyses",
    default: Any = None,
) -> Any:
    """
    Retrieve an attached contact analysis.

    Parameters
    ----------
    dock_model : DockModel
        Dock model.
    pose_identifier : str or None, optional
        Pose-analysis key. When omitted, the complete mapping is returned.
    attribute_name : str, optional
        Attribute containing contact analyses.
    default : Any, optional
        Fallback value.

    Returns
    -------
    Any
        Contact result, analysis mapping or ``default``.
    """

    analyses = _get_dock_model_value(
        dock_model,
        (
            attribute_name,
        ),
        default=None,
        call_if_callable=False,
    )

    if analyses is None:
        return default

    if pose_identifier is None:
        return analyses

    if isinstance(
        analyses,
        Mapping,
    ):
        return analyses.get(
            pose_identifier,
            default,
        )

    return default


# -----------------------------------------------------------------------------
# High-level contact analysis
# -----------------------------------------------------------------------------

def analyze_contacts(
    dock_model: DockModel,
    *,
    receptor: Optional[ReceptorLike] = None,
    pose: Optional[LigandLike] = None,
    pose_index: Optional[int] = None,
    cutoff: Optional[Number] = None,
    heavy_only: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    exclude_same_residue: bool = True,
    classify: bool = True,
    clash_overlap: Optional[Number] = None,
    close_tolerance: Optional[Number] = None,
    vdw_tolerance: Optional[Number] = None,
    default_vdw_radius: Optional[Number] = None,
    block_size: Optional[int] = None,
    maximum_matrix_elements: Optional[int] = None,
    maximum_contacts: Optional[int] = None,
    residue_side: str = "receptor",
    attach: bool = True,
    attach_to_pose: bool = True,
    overwrite: bool = True,
    analysis_attribute: str = "contact_analyses",
    metadata: Optional[Mapping[str, Any]] = None,
    scene: bool = True,
) -> ContactAnalysisResult:
    """Analyze contacts for one DockModel-compatible pose."""

    if dock_model is None:
        raise TypeError(
            "dock_model cannot be None."
        )

    normalized_side = _validate_contact_side(
        residue_side
    )
    scene_value = _validate_scene_flag(
        scene
    )

    resolved_receptor = (
        receptor
        if receptor is not None
        else get_dock_model_receptor(
            dock_model,
            default=None,
        )
    )

    if resolved_receptor is None:
        raise ValueError(
            "Could not resolve a receptor from dock_model. "
            "Provide receptor explicitly."
        )

    resolved_pose = (
        pose
        if pose is not None
        else get_dock_model_pose(
            dock_model,
            pose_index=pose_index,
            default=None,
        )
    )

    resolved_ligand = (
        pose
        if pose is not None
        else get_dock_model_ligand(
            dock_model,
            default=resolved_pose,
        )
    )

    if resolved_pose is None:
        resolved_pose = resolved_ligand

    if resolved_ligand is None:
        raise ValueError(
            "Could not resolve a ligand or pose from dock_model. "
            "Provide pose explicitly or use a valid pose_index."
        )

    resolved_cutoff = _resolve_contact_cutoff(
        cutoff
    )
    dock_identifier = get_dock_model_identifier(
        dock_model
    )
    pose_identifier = get_pose_identifier(
        resolved_pose,
        pose_index=pose_index,
    )

    analysis_metadata: Dict[str, Any] = (
        {}
        if metadata is None
        else dict(
            metadata
        )
    )
    analysis_metadata.setdefault(
        "dock_model_identifier",
        dock_identifier,
    )
    analysis_metadata.setdefault(
        "pose_identifier",
        pose_identifier,
    )
    analysis_metadata.setdefault(
        "pose_index",
        pose_index,
    )
    analysis_metadata.setdefault(
        "cutoff",
        float(
            resolved_cutoff
        ),
    )
    analysis_metadata.setdefault(
        "heavy_only",
        bool(
            heavy_only
        ),
    )
    analysis_metadata.setdefault(
        "exclude_solvent",
        bool(
            exclude_solvent
        ),
    )
    analysis_metadata.setdefault(
        "exclude_ions",
        bool(
            exclude_ions
        ),
    )
    analysis_metadata.setdefault(
        "exclude_same_residue",
        bool(
            exclude_same_residue
        ),
    )
    analysis_metadata.setdefault(
        "general_classification_applied",
        bool(
            classify
        ),
    )
    analysis_metadata.setdefault(
        "scene_coordinates",
        scene_value,
    )

    ligand_atoms, receptor_atoms = select_contact_collections(
        resolved_ligand,
        resolved_receptor,
        heavy_only=heavy_only,
        exclude_solvent=exclude_solvent,
        exclude_ions=exclude_ions,
        require_coordinate=True,
    )

    atom_contacts = find_atom_contacts(
        ligand_atoms,
        receptor_atoms,
        cutoff=resolved_cutoff,
        exclude_same_residue=exclude_same_residue,
        exclude_identical_atoms=True,
        block_size=block_size,
        maximum_matrix_elements=maximum_matrix_elements,
        sort_by_distance=True,
        maximum_contacts=maximum_contacts,
        classification=CONTACT_TYPE_CONTACT,
        metadata=analysis_metadata,
        scene=scene_value,
    )

    if classify:
        atom_contacts = classify_contacts(
            atom_contacts,
            clash_overlap=clash_overlap,
            close_tolerance=close_tolerance,
            vdw_tolerance=vdw_tolerance,
            default_radius=default_vdw_radius,
            sort_by_distance=True,
        )

    grouped_residue_contacts = residue_contacts(
        atom_contacts,
        side=normalized_side,
        include_missing=False,
        include_structure_identity=True,
        sort_contacts=True,
        sort_residues=True,
        metadata={
            "dock_model_identifier": dock_identifier,
            "pose_identifier": pose_identifier,
        },
    )

    statistics = contact_statistics(
        atom_contacts,
        residue_side=normalized_side,
        include_residue_statistics=True,
        include_classification_statistics=True,
    )
    statistics[
        "ligand_atom_count"
    ] = len(
        ligand_atoms
    )
    statistics[
        "receptor_atom_count"
    ] = len(
        receptor_atoms
    )
    statistics[
        "pose_identifier"
    ] = pose_identifier
    statistics[
        "dock_model_identifier"
    ] = dock_identifier
    statistics[
        "scene_coordinates"
    ] = scene_value

    result = ContactAnalysisResult(
        contacts=atom_contacts,
        residue_contacts=grouped_residue_contacts,
        ligand_atoms=ligand_atoms,
        receptor_atoms=receptor_atoms,
        cutoff=resolved_cutoff,
        statistics=statistics,
        metadata=analysis_metadata,
    )

    attachment_successful = False

    if attach:
        attachment_successful = attach_contact_analysis(
            dock_model,
            result,
            pose=resolved_pose,
            pose_identifier=pose_identifier,
            attribute_name=analysis_attribute,
            attach_to_pose=attach_to_pose,
            overwrite=overwrite,
        )

    if attach and not attachment_successful:
        try:
            _LOGGER.warning(
                "Contact analysis completed, but the result "
                "could not be attached to DockModel or pose."
            )
        except Exception:
            pass

    return result



# -----------------------------------------------------------------------------
# Multiple-pose analysis
# -----------------------------------------------------------------------------

def analyze_all_pose_contacts(
    dock_model: DockModel,
    *,
    receptor: Optional[ReceptorLike] = None,
    poses: Optional[Iterable[LigandLike]] = None,
    cutoff: Optional[Number] = None,
    heavy_only: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    exclude_same_residue: bool = True,
    classify: bool = True,
    block_size: Optional[int] = None,
    maximum_matrix_elements: Optional[int] = None,
    attach: bool = True,
    overwrite: bool = True,
    analysis_attribute: str = "contact_analyses",
    scene: bool = True,
) -> Tuple[ContactAnalysisResult, ...]:
    """Analyze all poses exposed by a DockModel-compatible object."""

    scene_value = _validate_scene_flag(
        scene
    )

    if poses is None:
        resolved_poses = _get_dock_model_value(
            dock_model,
            (
                "poses",
                "pose_models",
                "ligand_poses",
                "conformations",
            ),
            default=None,
            call_if_callable=True,
        )

        if resolved_poses is None:
            return (
                analyze_contacts(
                    dock_model,
                    receptor=receptor,
                    pose=None,
                    pose_index=0,
                    cutoff=cutoff,
                    heavy_only=heavy_only,
                    exclude_solvent=exclude_solvent,
                    exclude_ions=exclude_ions,
                    exclude_same_residue=exclude_same_residue,
                    classify=classify,
                    block_size=block_size,
                    maximum_matrix_elements=maximum_matrix_elements,
                    attach=attach,
                    attach_to_pose=True,
                    overwrite=overwrite,
                    analysis_attribute=analysis_attribute,
                    scene=scene_value,
                ),
            )

        if isinstance(
            resolved_poses,
            Mapping,
        ):
            normalized_poses = tuple(
                resolved_poses.values()
            )
        else:
            try:
                normalized_poses = tuple(
                    resolved_poses
                )
            except TypeError:
                normalized_poses = (
                    resolved_poses,
                )
    else:
        if isinstance(
            poses,
            (
                str,
                bytes,
                Mapping,
            ),
        ):
            raise TypeError(
                "poses must be an iterable of pose-like objects."
            )
        try:
            normalized_poses = tuple(
                poses
            )
        except TypeError as error:
            raise TypeError(
                "poses must be iterable."
            ) from error

    if not normalized_poses:
        return ()

    results: List[ContactAnalysisResult] = []

    for pose_index, pose in enumerate(
        normalized_poses
    ):
        results.append(
            analyze_contacts(
                dock_model,
                receptor=receptor,
                pose=pose,
                pose_index=pose_index,
                cutoff=cutoff,
                heavy_only=heavy_only,
                exclude_solvent=exclude_solvent,
                exclude_ions=exclude_ions,
                exclude_same_residue=exclude_same_residue,
                classify=classify,
                block_size=block_size,
                maximum_matrix_elements=maximum_matrix_elements,
                attach=attach,
                attach_to_pose=True,
                overwrite=overwrite,
                analysis_attribute=analysis_attribute,
                scene=scene_value,
            )
        )

    return tuple(
        results
    )



# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES = [
    "get_dock_model_receptor",
    "get_dock_model_ligand",
    "get_dock_model_pose",
    "get_dock_model_identifier",
    "get_pose_identifier",
    "attach_contact_analysis",
    "get_attached_contact_analysis",
    "analyze_contacts",
    "analyze_all_pose_contacts",
]

_register_public_names(_SECTION_10_PUBLIC_NAMES)


# =============================================================================
# End of Section 10
# =============================================================================


# =============================================================================
# Section 11 — Self-tests
# =============================================================================


# -----------------------------------------------------------------------------
# Synthetic test objects
# -----------------------------------------------------------------------------

@dataclass
class _TestElement:
    """Minimal synthetic chemical element used by self-tests."""

    name: str
    number: int


@dataclass
class _TestResidue:
    """Minimal synthetic residue used by self-tests."""

    name: str
    number: int
    chain_id: str = "A"


@dataclass
class _TestAtom:
    """Minimal synthetic atom used by self-tests."""

    name: str
    element: _TestElement
    coord: FloatArray
    residue: _TestResidue
    structure: Any = None
    scene_coord: Optional[FloatArray] = None


@dataclass
class _TestStructure:
    """Minimal synthetic molecular structure used by self-tests."""

    name: str
    atoms: Sequence[_TestAtom] = field(
        default_factory=tuple
    )

    def __post_init__(
        self,
    ) -> None:
        """Associate atoms with this structure."""

        self.atoms = tuple(
            self.atoms
        )

        for atom in self.atoms:
            atom.structure = self


@dataclass
class _TestDockModel:
    """Minimal synthetic dock model used by self-tests."""

    receptor: _TestStructure
    poses: Sequence[_TestStructure]
    name: str = "synthetic_dock_model"
    contact_analyses: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )
    contacts: List[Any] = field(
        default_factory=list
    )
    statistics: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """Normalize pose collection."""

        self.poses = tuple(
            self.poses
        )

    def update_statistics(
        self,
        additional_statistics: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update DockModel-compatible contact statistics."""

        self.statistics[
            "contacts"
        ] = len(
            self.contacts
        )

        if additional_statistics:
            self.statistics.update(
                additional_statistics
            )

        return self.statistics


@dataclass
class _TestSinglePoseDockModel:
    """Repository-style DockModel fixture for integration tests."""

    name: str
    receptor: _TestStructure
    pose: _TestStructure
    ligand: Any
    contacts: List[Any] = field(
        default_factory=list
    )
    statistics: Dict[str, Any] = field(
        default_factory=dict
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )
    contact_analyses: Dict[str, Any] = field(
        default_factory=dict
    )

    def update_statistics(
        self,
        additional_statistics: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.statistics[
            "contacts"
        ] = len(
            self.contacts
        )
        self.statistics[
            "total_interactions"
        ] = len(
            self.contacts
        )

        if additional_statistics:
            self.statistics.update(
                additional_statistics
            )

        return self.statistics

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.update_statistics()

        return {
            "name": self.name,
            "contacts": [
                contact.to_dict()
                if hasattr(
                    contact,
                    "to_dict",
                )
                else contact
                for contact in self.contacts
            ],
            "statistics": dict(
                self.statistics
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Synthetic system construction
# -----------------------------------------------------------------------------

def _make_test_atom(
    name: str,
    element_symbol: str,
    atomic_number: int,
    coordinate: Sequence[Number],
    residue: _TestResidue,
    scene_coordinate: Optional[Sequence[Number]] = None,
) -> _TestAtom:
    """Construct a synthetic atom."""

    return _TestAtom(
        name=name,
        element=_TestElement(
            name=element_symbol,
            number=atomic_number,
        ),
        coord=np.asarray(
            coordinate,
            dtype=np.float64,
        ),
        residue=residue,
        scene_coord=np.asarray(
            coordinate
            if scene_coordinate is None
            else scene_coordinate,
            dtype=np.float64,
        ),
    )


def _build_synthetic_contact_system(
) -> Tuple[
    _TestStructure,
    _TestStructure,
    _TestDockModel,
]:
    """Build a deterministic receptor-ligand test system."""

    ligand_residue = _TestResidue(
        name="LIG",
        number=1,
        chain_id="L",
    )

    tyr58 = _TestResidue(
        name="TYR",
        number=58,
        chain_id="A",
    )

    phe77 = _TestResidue(
        name="PHE",
        number=77,
        chain_id="A",
    )

    ser205 = _TestResidue(
        name="SER",
        number=205,
        chain_id="A",
    )

    ligand_atoms = (
        _make_test_atom(
            "C1",
            "C",
            6,
            (
                0.0,
                0.0,
                0.0,
            ),
            ligand_residue,
        ),
        _make_test_atom(
            "O1",
            "O",
            8,
            (
                0.0,
                3.0,
                0.0,
            ),
            ligand_residue,
        ),
    )

    receptor_atoms = (
        # C-C distance = 3.40 Å.
        # vdW sum = 3.40 Å -> close contact.
        _make_test_atom(
            "CZ",
            "C",
            6,
            (
                3.40,
                0.0,
                0.0,
            ),
            tyr58,
        ),
        # C-O distance = 2.80 Å.
        # vdW sum = 3.22 Å -> overlap = 0.42 Å -> clash.
        _make_test_atom(
            "OH",
            "O",
            8,
            (
                2.80,
                0.0,
                0.0,
            ),
            tyr58,
        ),
        # O-N distance = 3.20 Å.
        # vdW sum = 3.07 Å -> deviation = 0.13 Å -> close.
        _make_test_atom(
            "N",
            "N",
            7,
            (
                0.0,
                6.20,
                0.0,
            ),
            phe77,
        ),
        # O-C distance = 3.50 Å.
        # vdW sum = 3.22 Å -> deviation = 0.28 Å -> vdW.
        _make_test_atom(
            "CB",
            "C",
            6,
            (
                0.0,
                6.50,
                0.0,
            ),
            ser205,
        ),
    )

    ligand = _TestStructure(
        name="pose_0",
        atoms=ligand_atoms,
    )

    receptor = _TestStructure(
        name="receptor",
        atoms=receptor_atoms,
    )

    dock_model = _TestDockModel(
        receptor=receptor,
        poses=(
            ligand,
        ),
    )

    return (
        ligand,
        receptor,
        dock_model,
    )


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------

def _assert_close(
    observed: Number,
    expected: Number,
    *,
    tolerance: Number = 1.0e-8,
    message: str = "",
) -> None:
    """Assert that two numeric values are approximately equal."""

    observed_value = np.float64(
        observed
    )

    expected_value = np.float64(
        expected
    )

    tolerance_value = np.float64(
        tolerance
    )

    if not np.isclose(
        observed_value,
        expected_value,
        atol=tolerance_value,
        rtol=np.float64(0.0),
    ):
        default_message = (
            f"Expected {expected_value}, "
            f"observed {observed_value}."
        )

        raise AssertionError(
            message
            or default_message
        )


def _contact_signature(
    contacts: Iterable[AtomContact],
) -> Tuple[
    Tuple[
        int,
        int,
        float,
    ],
    ...,
]:
    """Build a deterministic contact signature."""

    normalized_contacts = _validate_atom_contacts(
        contacts,
        allow_empty=True,
    )

    return tuple(
        (
            (
                -1
                if contact.atom_1_index is None
                else contact.atom_1_index
            ),
            (
                -1
                if contact.atom_2_index is None
                else contact.atom_2_index
            ),
            round(
                float(
                    contact.distance
                ),
                8,
            ),
        )
        for contact in normalized_contacts
    )


def _require(
    condition: Any,
    message: str,
) -> None:
    """Raise ``AssertionError`` when a self-test condition is false."""

    if not bool(
        condition
    ):
        raise AssertionError(
            message
        )


# -----------------------------------------------------------------------------
# Individual self-tests
# -----------------------------------------------------------------------------

def _test_public_api(
) -> None:
    """Validate exported names and downstream import contracts."""

    _require(
        all(
            isinstance(
                name,
                str,
            )
            and bool(
                name
            )
            for name in __all__
        ),
        "__all__ must contain only non-empty strings.",
    )

    _require(
        len(
            __all__
        ) == len(
            set(
                __all__
            )
        ),
        "__all__ contains duplicate names.",
    )

    module_globals = globals()

    missing_exports = tuple(
        name
        for name in __all__
        if name not in module_globals
    )

    _require(
        not missing_exports,
        "__all__ contains undefined names: "
        f"{missing_exports!r}.",
    )

    downstream_names = (
        "AtomContact",
        "ContactAnalysisResult",
        "ResidueContact",
        "ResidueContactKey",
        "atom_coordinates",
        "filter_atoms",
        "get_atom_atomic_number",
        "get_atom_coordinate",
        "get_atom_element",
        "get_atom_identifier",
        "get_atom_index",
        "get_atom_name",
        "get_atom_residue",
        "get_atom_structure",
        "get_dock_model_identifier",
        "get_dock_model_pose",
        "get_dock_model_receptor",
        "get_pose_identifier",
        "get_residue_contact_key",
        "is_atom_like",
        "is_heavy_atom",
        "is_hydrogen_atom",
        "select_contact_collections",
        "validate_atom",
        "validate_atom_collection",
    )

    missing_downstream_names = tuple(
        name
        for name in downstream_names
        if name not in __all__
        or name not in module_globals
    )

    _require(
        not missing_downstream_names,
        "Required downstream API names are unavailable: "
        f"{missing_downstream_names!r}.",
    )


def _test_atom_helpers(
) -> None:
    """Test synthetic atom identification and coordinate helpers."""

    ligand, _, _ = (
        _build_synthetic_contact_system()
    )

    atom = ligand.atoms[
        0
    ]

    _require(
        is_atom_like(
            atom
        ),
        "Self-test assertion failed at source line 11061.",
    )

    validate_atom(
        atom,
        require_coordinate=True,
    )

    _require(
        get_atom_name(
            atom
        ) == "C1",
        "Self-test assertion failed at source line 11070.",
    )

    _require(
        get_atom_element(
            atom
        ) == "C",
        "Self-test assertion failed at source line 11074.",
    )

    _require(
        get_atom_atomic_number(
            atom
        ) == 6,
        "Self-test assertion failed at source line 11078.",
    )

    _require(
        get_atom_residue(
            atom
        ) is atom.residue,
        "Self-test assertion failed at source line 11082.",
    )

    _require(
        get_atom_structure(
            atom
        ) is ligand,
        "Self-test assertion failed at source line 11086.",
    )

    coordinate = get_atom_coordinate(
        atom
    )

    _require(
        coordinate.shape == (
            3,
        ),
        "Self-test assertion failed at source line 11094.",
    )

    _require(
        coordinate.dtype == np.float64,
        "Self-test assertion failed at source line 11098.",
    )

    _require(
        is_heavy_atom(
            atom
        ),
        "Self-test assertion failed at source line 11100.",
    )

    _require(
        not is_hydrogen_atom(
            atom
        ),
        "Self-test assertion failed at source line 11104.",
    )


def _test_collection_selection(
) -> None:
    """Test ligand and receptor atom selection."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    ligand_atoms, receptor_atoms = (
        select_contact_collections(
            ligand,
            receptor,
            heavy_only=True,
            exclude_solvent=True,
            exclude_ions=True,
            require_coordinate=True,
        )
    )

    _require(
        len(
            ligand_atoms
        ) == 2,
        "Self-test assertion failed at source line 11128.",
    )

    _require(
        len(
            receptor_atoms
        ) == 4,
        "Self-test assertion failed at source line 11132.",
    )

    _require(
        all(
            is_heavy_atom(
                atom
            )
            for atom in ligand_atoms
        ),
        "Self-test assertion failed at source line 11136.",
    )

    _require(
        all(
            is_heavy_atom(
                atom
            )
            for atom in receptor_atoms
        ),
        "Self-test assertion failed at source line 11143.",
    )


def _test_contact_search(
) -> None:
    """Test full-matrix and blocked contact searching."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    full_matrix_contacts = (
        find_atom_contacts(
            ligand.atoms,
            receptor.atoms,
            cutoff=4.0,
            block_size=None,
            maximum_matrix_elements=1_000_000,
        )
    )

    blocked_contacts = find_atom_contacts(
        ligand.atoms,
        receptor.atoms,
        cutoff=4.0,
        block_size=1,
        maximum_matrix_elements=1,
    )

    _require(
        len(
            full_matrix_contacts
        ) == 4,
        "Self-test assertion failed at source line 11177.",
    )

    _require(
        len(
            blocked_contacts
        ) == 4,
        "Self-test assertion failed at source line 11181.",
    )

    _require(
        _contact_signature(
                full_matrix_contacts
            )
            == _contact_signature(
                blocked_contacts
            ),
        "Self-test assertion failed at source line 11185.",
    )

    expected_distances = (
        2.80,
        3.20,
        3.40,
        3.50,
    )

    observed_distances = tuple(
        sorted(
            float(
                contact.distance
            )
            for contact in full_matrix_contacts
        )
    )

    for observed, expected in zip(
        observed_distances,
        expected_distances,
    ):
        _assert_close(
            observed,
            expected,
        )


def _test_coordinate_extraction_efficiency() -> None:
    """Ensure contact search reads each coordinate only once."""

    class CountingAtom:
        def __init__(self, coordinate: Sequence[Number], name: str) -> None:
            self.name = name
            self.element = _TestElement("C", 6)
            self.residue = _TestResidue("LIG", 1, "A")
            self._coordinate = np.asarray(coordinate, dtype=np.float64)
            self.coordinate_reads = 0

        @property
        def coord(self) -> FloatArray:
            self.coordinate_reads += 1
            return self._coordinate

    atom_1 = CountingAtom((0.0, 0.0, 0.0), "C1")
    atom_2 = CountingAtom((3.0, 0.0, 0.0), "C2")
    result = find_atom_contacts((atom_1,), (atom_2,), cutoff=4.0, scene=False)
    _require(len(result) == 1, "Expected one counting-atom contact.")
    _require(atom_1.coordinate_reads == 1, "First coordinate was read repeatedly.")
    _require(atom_2.coordinate_reads == 1, "Second coordinate was read repeatedly.")


def _test_closest_contact(
) -> None:
    """Test closest-contact detection."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    result = closest_contact(
        ligand.atoms,
        receptor.atoms,
        cutoff=4.0,
        block_size=1,
    )

    _require(
        result is not None,
        "Self-test assertion failed at source line 11235.",
    )

    _assert_close(
        result.distance,
        2.80,
    )

    _require(
        result.atom_1 is ligand.atoms[
            0
        ],
        "Self-test assertion failed at source line 11242.",
    )

    _require(
        result.atom_2 is receptor.atoms[
            1
        ],
        "Self-test assertion failed at source line 11246.",
    )

    _require(
        result.is_contact,
        "Self-test assertion failed at source line 11250.",
    )


def _test_residue_grouping(
) -> None:
    """Test grouping and summaries by receptor residue."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    contacts = find_contacts(
        ligand,
        receptor,
        cutoff=4.0,
    )

    groups = group_contacts_by_residue(
        contacts,
        side="receptor",
    )

    _require(
        len(
            groups
        ) == 3,
        "Self-test assertion failed at source line 11272.",
    )

    _require(
        len(
            groups[
                (
                    "TYR",
                    58,
                    "A",
                )
            ]
        ) == 2,
        "Self-test assertion failed at source line 11276.",
    )

    _require(
        len(
            groups[
                (
                    "PHE",
                    77,
                    "A",
                )
            ]
        ) == 1,
        "Self-test assertion failed at source line 11286.",
    )

    _require(
        len(
            groups[
                (
                    "SER",
                    205,
                    "A",
                )
            ]
        ) == 1,
        "Self-test assertion failed at source line 11296.",
    )

    results = residue_contacts(
        contacts,
        side="receptor",
    )

    _require(
        len(
            results
        ) == 3,
        "Self-test assertion failed at source line 11311.",
    )

    counts = residue_contact_counts(
        contacts,
        side="receptor",
    )

    _require(
        counts[
            (
                "TYR",
                58,
                "A",
            )
        ] == 2,
        "Self-test assertion failed at source line 11320.",
    )

    residue_keys = contacting_residues(
        contacts,
        side="receptor",
        return_keys=True,
    )

    _require(
        residue_keys == (
            (
                "TYR",
                58,
                "A",
            ),
            (
                "PHE",
                77,
                "A",
            ),
            (
                "SER",
                205,
                "A",
            ),
        ),
        "Self-test assertion failed at source line 11334.",
    )

    summary = summarize_residue_contacts(
        contacts,
        side="receptor",
        include_chain=False,
    )

    _require(
        summary[
            0
        ] == (
            "TYR58",
            2,
        ),
        "Self-test assertion failed at source line 11358.",
    )


def _test_contact_classification(
) -> None:
    """Test general geometric contact classification."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    contacts = find_contacts(
        ligand,
        receptor,
        cutoff=4.0,
    )

    classified = classify_contacts(
        contacts
    )

    counts = contact_classification_counts(
        classified
    )

    _require(
        counts[
            CONTACT_TYPE_CLASH
        ] == 1,
        "Self-test assertion failed at source line 11388.",
    )

    _require(
        counts[
            CONTACT_TYPE_CLOSE_CONTACT
        ] == 2,
        "Self-test assertion failed at source line 11392.",
    )

    _require(
        counts[
            CONTACT_TYPE_VDW
        ] == 1,
        "Self-test assertion failed at source line 11396.",
    )

    _require(
        counts[
            CONTACT_TYPE_UNKNOWN
        ] == 0,
        "Self-test assertion failed at source line 11400.",
    )

    clashes = contacts_by_classification(
        classified,
        CONTACT_TYPE_CLASH,
    )

    _require(
        len(
            clashes
        ) == 1,
        "Self-test assertion failed at source line 11409.",
    )

    _assert_close(
        clashes[
            0
        ].distance,
        2.80,
    )


def _test_contact_statistics(
) -> None:
    """Test descriptive statistics and distance distributions."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    contacts = classify_contacts(
        find_contacts(
            ligand,
            receptor,
            cutoff=4.0,
        )
    )

    statistics = contact_statistics(
        contacts,
        residue_side="receptor",
    )

    _require(
        statistics[
            "contact_count"
        ] == 4,
        "Self-test assertion failed at source line 11442.",
    )

    _require(
        statistics[
            "unique_atom_1_count"
        ] == 2,
        "Self-test assertion failed at source line 11446.",
    )

    _require(
        statistics[
            "unique_atom_2_count"
        ] == 4,
        "Self-test assertion failed at source line 11450.",
    )

    _require(
        statistics[
            "residues"
        ][
            "residue_count"
        ] == 3,
        "Self-test assertion failed at source line 11454.",
    )

    _assert_close(
        statistics[
            "distance"
        ][
            "minimum"
        ],
        2.80,
    )

    _assert_close(
        statistics[
            "distance"
        ][
            "maximum"
        ],
        3.50,
    )

    _assert_close(
        statistics[
            "distance"
        ][
            "mean"
        ],
        3.225,
    )

    distribution = (
        contact_distance_distribution(
            contacts,
            bins=2,
            distance_range=(
                2.5,
                3.6,
            ),
        )
    )

    _require(
        distribution[
            "sample_count"
        ] == 4,
        "Self-test assertion failed at source line 11498.",
    )

    _require(
        distribution[
            "bin_count"
        ] == 2,
        "Self-test assertion failed at source line 11502.",
    )

    _require(
        sum(
            distribution[
                "counts"
            ]
        ) == 4,
        "Self-test assertion failed at source line 11506.",
    )

    _require(
        distribution[
            "cumulative_counts"
        ][
            -1
        ] == 4,
        "Self-test assertion failed at source line 11512.",
    )

    summary = summarize_contacts(
        contacts,
        residue_side="receptor",
        include_chain=False,
    )

    _require(
        summary[
            "contact_count"
        ] == 4,
        "Self-test assertion failed at source line 11524.",
    )

    _require(
        summary[
            "contacting_residue_count"
        ] == 3,
        "Self-test assertion failed at source line 11528.",
    )

    _require(
        summary[
            "residue_contacts"
        ][
            0
        ][
            "residue"
        ] == "TYR58",
        "Self-test assertion failed at source line 11532.",
    )

    text_summary = format_contact_summary(
        contacts,
        include_chain=False,
    )

    _require(
        "Contacts: 4" in text_summary,
        "Self-test assertion failed at source line 11545.",
    )
    _require(
        "TYR58" in text_summary,
        "Self-test assertion failed at source line 11546.",
    )


def _test_empty_contact_statistics(
) -> None:
    """Test statistics for an empty contact collection."""

    statistics = contact_statistics(
        ()
    )

    _require(
        statistics[
            "contact_count"
        ] == 0,
        "Self-test assertion failed at source line 11557.",
    )

    _require(
        not statistics[
            "has_contacts"
        ],
        "Self-test assertion failed at source line 11561.",
    )

    _require(
        statistics[
            "distance"
        ][
            "minimum"
        ] is None,
        "Self-test assertion failed at source line 11565.",
    )

    distribution = (
        contact_distance_distribution(
            (),
            bins=4,
            distance_range=(
                0.0,
                4.0,
            ),
        )
    )

    _require(
        distribution[
            "sample_count"
        ] == 0,
        "Self-test assertion failed at source line 11582.",
    )

    _require(
        distribution[
            "bin_count"
        ] == 4,
        "Self-test assertion failed at source line 11586.",
    )


def _test_dock_model_integration(
) -> None:
    """Test high-level DockModel contact analysis."""

    _, _, dock_model = (
        _build_synthetic_contact_system()
    )

    result = analyze_contacts(
        dock_model,
        pose_index=0,
        cutoff=4.0,
        classify=True,
        attach=True,
        attach_to_pose=True,
    )

    _require(
        isinstance(
            result,
            ContactAnalysisResult,
        ),
        "Self-test assertion failed at source line 11608.",
    )

    _require(
        result.contact_count == 4,
        "Self-test assertion failed at source line 11613.",
    )
    _require(
        result.residue_count == 3,
        "Self-test assertion failed at source line 11614.",
    )
    _require(
        result.ligand_atom_count == 2,
        "Self-test assertion failed at source line 11615.",
    )
    _require(
        result.receptor_atom_count == 4,
        "Self-test assertion failed at source line 11616.",
    )
    _require(
        result.has_contacts,
        "Self-test assertion failed at source line 11617.",
    )
    _require(
        result.has_clashes,
        "Self-test assertion failed at source line 11618.",
    )

    _require(
        "pose_0" in (
            dock_model.contact_analyses
        ),
        "Self-test assertion failed at source line 11620.",
    )

    _require(
        dock_model.contact_analyses[
                "pose_0"
            ]
            is result,
        "Self-test assertion failed at source line 11624.",
    )

    pose = dock_model.poses[
        0
    ]

    _require(
        getattr(
            pose,
            "contact_analysis",
            None,
        ) is result,
        "Self-test assertion failed at source line 11635.",
    )

    _require(
        getattr(
            pose,
            "contacts",
            None,
        ) == result.contacts,
        "Self-test assertion failed at source line 11641.",
    )

    attached_result = (
        get_attached_contact_analysis(
            dock_model,
            pose_identifier="pose_0",
        )
    )

    _require(
        attached_result is result,
        "Self-test assertion failed at source line 11654.",
    )


def _test_all_pose_integration(
) -> None:
    """Test analysis of multiple synthetic poses."""

    ligand, receptor, _ = (
        _build_synthetic_contact_system()
    )

    second_pose_atoms = tuple(
        _TestAtom(
            name=atom.name,
            element=_TestElement(
                name=atom.element.name,
                number=atom.element.number,
            ),
            coord=np.asarray(
                atom.coord,
                dtype=np.float64,
            )
            + np.asarray(
                (
                    20.0,
                    0.0,
                    0.0,
                ),
                dtype=np.float64,
            ),
            residue=atom.residue,
        )
        for atom in ligand.atoms
    )

    second_pose = _TestStructure(
        name="pose_1",
        atoms=second_pose_atoms,
    )

    dock_model = _TestDockModel(
        receptor=receptor,
        poses=(
            ligand,
            second_pose,
        ),
    )

    results = analyze_all_pose_contacts(
        dock_model,
        cutoff=4.0,
        attach=True,
    )

    _require(
        len(
            results
        ) == 2,
        "Self-test assertion failed at source line 11708.",
    )

    _require(
        results[
            0
        ].contact_count == 4,
        "Self-test assertion failed at source line 11712.",
    )

    _require(
        results[
            1
        ].contact_count == 0,
        "Self-test assertion failed at source line 11716.",
    )

    _require(
        "pose_0" in (
            dock_model.contact_analyses
        ),
        "Self-test assertion failed at source line 11720.",
    )

    _require(
        "pose_1" in (
            dock_model.contact_analyses
        ),
        "Self-test assertion failed at source line 11724.",
    )


def _test_scene_coordinate_integration(
) -> None:
    """Test local versus ChimeraX scene-coordinate contact searches."""

    ligand_residue = _TestResidue(
        "LIG",
        1,
        "L",
    )
    receptor_residue = _TestResidue(
        "TYR",
        58,
        "A",
    )

    ligand_atom = _make_test_atom(
        "C1",
        "C",
        6,
        (
            100.0,
            0.0,
            0.0,
        ),
        ligand_residue,
        scene_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
    )
    receptor_atom = _make_test_atom(
        "CZ",
        "C",
        6,
        (
            0.0,
            0.0,
            0.0,
        ),
        receptor_residue,
        scene_coordinate=(
            3.4,
            0.0,
            0.0,
        ),
    )

    scene_contacts = find_atom_contacts(
        (
            ligand_atom,
        ),
        (
            receptor_atom,
        ),
        cutoff=4.0,
    )
    local_contacts = find_atom_contacts(
        (
            ligand_atom,
        ),
        (
            receptor_atom,
        ),
        cutoff=4.0,
        scene=False,
    )

    _require(
        len(
            scene_contacts
        ) == 1,
        "Scene-coordinate contact detection failed.",
    )
    _require(
        len(
            local_contacts
        ) == 0,
        "Local-coordinate search ignored scene=False.",
    )
    _assert_close(
        scene_contacts[
            0
        ].distance,
        3.4,
    )


def _test_repository_dock_model_integration(
) -> None:
    """Test attachment and serialization with the repository DockModel shape."""

    ligand, receptor, _ = _build_synthetic_contact_system()
    dock_model = _TestSinglePoseDockModel(
        name="pose_0",
        receptor=receptor,
        pose=ligand,
        ligand=ligand,
    )

    result = analyze_contacts(
        dock_model,
        cutoff=4.0,
        attach=True,
    )

    _require(
        result.contact_count == 4,
        "Repository-style DockModel analysis returned the wrong count.",
    )
    _require(
        len(
            dock_model.contacts
        ) == 4,
        "DockModel.contacts was not populated.",
    )
    _require(
        dock_model.statistics.get(
            "contacts"
        ) == 4,
        "DockModel contact statistics were not synchronized.",
    )
    _require(
        dock_model.contact_analyses.get(
            "pose_0"
        ) is result,
        "The detailed contact analysis was not attached.",
    )

    import json

    json.dumps(
        result.to_dict()
    )
    json.dumps(
        dock_model.to_dict()
    )


def _test_invalid_integration_inputs(
) -> None:
    """Test empty, invalid and malformed integration inputs."""

    ligand, receptor, _ = _build_synthetic_contact_system()

    invalid_cases = (
        (
            ValueError,
            lambda: find_atom_contacts(
                (),
                receptor.atoms,
                cutoff=4.0,
            ),
        ),
        (
            ValueError,
            lambda: find_atom_contacts(
                ligand.atoms,
                receptor.atoms,
                cutoff=np.nan,
            ),
        ),
        (
            TypeError,
            lambda: find_atom_contacts(
                ligand.atoms,
                receptor.atoms,
                cutoff=4.0,
                scene="yes",
            ),
        ),
        (
            TypeError,
            lambda: analyze_contacts(
                None,
            ),
        ),
        (
            ValueError,
            lambda: analyze_contacts(
                {},
            ),
        ),
    )

    for expected_error, operation in invalid_cases:
        try:
            operation()
        except expected_error:
            continue
        except Exception as error:
            raise AssertionError(
                "An invalid input raised the wrong exception type."
            ) from error
        raise AssertionError(
            "An invalid integration input was accepted."
        )


# -----------------------------------------------------------------------------
# Self-test runner
# -----------------------------------------------------------------------------

_SELF_TEST_CODE_FAILURE = "code failure"
_SELF_TEST_TEST_FAILURE = "test failure"
_SELF_TEST_ENVIRONMENTAL_LIMITATION = "environmental limitation"


def _classify_self_test_failure(
    error: BaseException,
) -> str:
    """Classify a self-test failure by its most likely source."""

    if isinstance(
        error,
        (
            ImportError,
            ModuleNotFoundError,
        ),
    ):
        return (
            _SELF_TEST_ENVIRONMENTAL_LIMITATION
        )

    traceback_object = error.__traceback__

    while (
        traceback_object is not None
        and traceback_object.tb_next is not None
    ):
        traceback_object = (
            traceback_object.tb_next
        )

    source_function = (
        ""
        if traceback_object is None
        else traceback_object.tb_frame.f_code.co_name
    )

    if (
        isinstance(
            error,
            AssertionError,
        )
        or not source_function.startswith(
            "_test_"
        )
    ):
        return _SELF_TEST_CODE_FAILURE

    return _SELF_TEST_TEST_FAILURE


def run_self_tests(
    *,
    verbose: bool = True,
    raise_on_failure: bool = True,
) -> bool:
    """
    Run the ``contacts.py`` self-test suite.

    Parameters
    ----------
    verbose : bool, optional
        Whether individual test results should be printed.
    raise_on_failure : bool, optional
        Whether the completed suite should raise when failures exist.

    Returns
    -------
    bool
        ``True`` when every test passes, otherwise ``False``.

    Raises
    ------
    AssertionError
        If failures exist and ``raise_on_failure=True``.

    Notes
    -----
    Tests use synthetic Python objects and NumPy arrays. ChimeraX is not
    required. Failures are classified as code failures, test failures or
    environmental limitations.
    """

    import traceback

    tests: Tuple[
        Tuple[
            str,
            Callable[
                [],
                None,
            ],
        ],
        ...,
    ] = (
        (
            "public API",
            _test_public_api,
        ),
        (
            "atom helpers",
            _test_atom_helpers,
        ),
        (
            "collection selection",
            _test_collection_selection,
        ),
        (
            "contact search",
            _test_contact_search,
        ),
        (
            "coordinate extraction efficiency",
            _test_coordinate_extraction_efficiency,
        ),
        (
            "closest contact",
            _test_closest_contact,
        ),
        (
            "residue grouping",
            _test_residue_grouping,
        ),
        (
            "contact classification",
            _test_contact_classification,
        ),
        (
            "contact statistics",
            _test_contact_statistics,
        ),
        (
            "empty statistics",
            _test_empty_contact_statistics,
        ),
        (
            "DockModel integration",
            _test_dock_model_integration,
        ),
        (
            "all-pose integration",
            _test_all_pose_integration,
        ),
        (
            "scene-coordinate integration",
            _test_scene_coordinate_integration,
        ),
        (
            "repository DockModel integration",
            _test_repository_dock_model_integration,
        ),
        (
            "invalid integration inputs",
            _test_invalid_integration_inputs,
        ),
    )

    passed = 0

    failures: List[
        Tuple[
            str,
            str,
            BaseException,
            str,
        ]
    ] = []

    category_counts = {
        _SELF_TEST_CODE_FAILURE: 0,
        _SELF_TEST_TEST_FAILURE: 0,
        _SELF_TEST_ENVIRONMENTAL_LIMITATION: 0,
    }

    if verbose:
        print(
            "=" * 72
        )
        print(
            "DockAnalyzer contacts.py self-tests"
        )
        print(
            "=" * 72
        )

    for test_name, test_function in tests:
        try:
            test_function()

        except Exception as error:
            failure_category = (
                _classify_self_test_failure(
                    error
                )
            )

            category_counts[
                failure_category
            ] += 1

            failure_traceback = (
                traceback.format_exc()
            )

            failures.append(
                (
                    test_name,
                    failure_category,
                    error,
                    failure_traceback,
                )
            )

            if verbose:
                print(
                    f"[FAIL] [{failure_category.upper()}] "
                    f"{test_name}: {type(error).__name__}: "
                    f"{error}"
                )

        else:
            passed += 1

            if verbose:
                print(
                    f"[PASS] {test_name}"
                )

    total = len(
        tests
    )

    failed = len(
        failures
    )

    if verbose:
        print(
            "-" * 72
        )
        print(
            f"Passed: {passed}/{total}"
        )
        print(
            f"Failed: {failed}/{total}"
        )

        if failures:
            print(
                "Failure categories: "
                f"code={category_counts[_SELF_TEST_CODE_FAILURE]}, "
                f"tests={category_counts[_SELF_TEST_TEST_FAILURE]}, "
                "environment="
                f"{category_counts[_SELF_TEST_ENVIRONMENTAL_LIMITATION]}"
            )

        print(
            "=" * 72
        )

    if failures and raise_on_failure:
        failure_details = "\n\n".join(
            (
                f"Test: {test_name}\n"
                f"Category: {failure_category}\n"
                f"{failure_traceback}"
            )
            for (
                test_name,
                failure_category,
                _,
                failure_traceback,
            ) in failures
        )

        raise AssertionError(
            "contacts.py self-tests failed:\n\n"
            f"{failure_details}"
        )

    return not failures


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_11_PUBLIC_NAMES = [
    "run_self_tests",
]

_register_public_names(_SECTION_11_PUBLIC_NAMES)


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_self_tests()


# =============================================================================
# End of Section 11
# =============================================================================




