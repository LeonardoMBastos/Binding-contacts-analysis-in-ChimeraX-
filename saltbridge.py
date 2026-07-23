# =============================================================================
# 1. IMPORTS E COMPATIBILIDADE
# =============================================================================

from __future__ import annotations

# =============================================================================
# Biblioteca padrão
# =============================================================================

import math
import json
import statistics
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# =============================================================================
# Compatibilidade opcional com NumPy
# =============================================================================

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None
    HAS_NUMPY = False

# =============================================================================
# Compatibilidade opcional com ChimeraX
# =============================================================================

try:  # pragma: no cover

    from chimerax.atomic import (
        Atom,
        Atoms,
        Residue,
        Structure,
    )

    HAS_CHIMERAX = True

except Exception:  # pragma: no cover

    Atom = Any
    Atoms = Any
    Residue = Any
    Structure = Any

    HAS_CHIMERAX = False

# =============================================================================
# Imports opcionais do DockAnalyzer
# =============================================================================

try:  # pragma: no cover

    from .dockmodel import DockModel

except Exception:  # pragma: no cover

    DockModel = Any

# =============================================================================
# Informações do módulo
# =============================================================================

__all__ = []

__author__ = "DockAnalyzer Project"
__version__ = "1.0.0"


# =============================================================================
# 2. CONSTANTES E ALIASES
# =============================================================================

# =============================================================================
# Tipos de interação
# =============================================================================

SALT_BRIDGE = "salt_bridge"

SALT_BRIDGE_TYPES = (
    "cation_anion",
    "anion_cation",
)

# =============================================================================
# Critérios geométricos padrão
# =============================================================================

DEFAULT_DISTANCE_CUTOFF = 4.0        # Å
DEFAULT_STRONG_CUTOFF = 3.2          # Å
DEFAULT_WEAK_CUTOFF = 4.0            # Å

DEFAULT_GROUP_RADIUS = 2.0           # Å

# =============================================================================
# Pesos utilizados pelo scoring
# =============================================================================

DEFAULT_SCORE_STRONG = 1.00
DEFAULT_SCORE_MODERATE = 0.75
DEFAULT_SCORE_WEAK = 0.50

# =============================================================================
# Cargas formais
# =============================================================================

FORMAL_POSITIVE = 1
FORMAL_NEGATIVE = -1
FORMAL_NEUTRAL = 0

# =============================================================================
# Elementos frequentemente encontrados
# =============================================================================

POSITIVE_ELEMENTS = frozenset({
    "N",
})

NEGATIVE_ELEMENTS = frozenset({
    "O",
    "S",
})

# =============================================================================
# Resíduos canônicos
# =============================================================================

CANONICAL_CATIONIC_RESIDUES = frozenset({
    "ARG",
    "LYS",
    "HIP",   # Histidina protonada
})

CANONICAL_ANIONIC_RESIDUES = frozenset({
    "ASP",
    "GLU",
})

# =============================================================================
# Átomos carregados conhecidos
# =============================================================================

CANONICAL_POSITIVE_ATOMS = {
    "ARG": frozenset({"NH1", "NH2", "NE"}),
    "LYS": frozenset({"NZ"}),
    "HIP": frozenset({"ND1", "NE2"}),
}

CANONICAL_NEGATIVE_ATOMS = {
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
}

# =============================================================================
# Classes de força
# =============================================================================

STRENGTH_STRONG = "strong"
STRENGTH_MODERATE = "moderate"
STRENGTH_WEAK = "weak"

STRENGTH_ORDER = (
    STRENGTH_STRONG,
    STRENGTH_MODERATE,
    STRENGTH_WEAK,
)

# =============================================================================
# Estado da interação
# =============================================================================

INTERACTION_VALID = "valid"
INTERACTION_INVALID = "invalid"

# =============================================================================
# Aliases de tipos
# =============================================================================

Coordinate = Tuple[float, float, float]

AtomLike = Union["Atom", Any]

ResidueLike = Union["Residue", Any]

StructureLike = Union["Structure", Any]



# =============================================================================
# 3. EXCEÇÕES
# =============================================================================


class SaltBridgeError(Exception):
    """
    Exceção-base para erros específicos do módulo ``saltbridge``.

    Todas as exceções próprias deste módulo devem herdar desta classe,
    permitindo que o chamador capture falhas relacionadas a pontes salinas
    sem interceptar exceções genéricas do restante do DockAnalyzer.
    """


class SaltBridgeConfigurationError(SaltBridgeError, ValueError):
    """
    Indica uma configuração inválida para a análise de pontes salinas.

    Exemplos:
        - cutoff de distância negativo;
        - limites de classificação em ordem incorreta;
        - pesos de scoring fora do intervalo permitido;
        - estratégia de reconhecimento desconhecida.
    """


class SaltBridgeRecognitionError(SaltBridgeError):
    """
    Indica falha durante o reconhecimento de átomos ou grupos carregados.

    Deve ser utilizada quando a estrutura recebida não puder ser interpretada
    com segurança, ou quando os dados químicos necessários estiverem
    inconsistentes.
    """


class ChargedGroupError(SaltBridgeRecognitionError):
    """
    Exceção-base para erros relacionados a grupos carregados.
    """


class InvalidChargedGroupError(ChargedGroupError, ValueError):
    """
    Indica que um grupo carregado não satisfaz os critérios mínimos.

    Exemplos:
        - grupo sem átomos;
        - polaridade inválida;
        - carga incompatível com a polaridade;
        - coordenadas ausentes;
        - átomos pertencentes a resíduos incompatíveis.
    """


class UnsupportedChargeError(ChargedGroupError):
    """
    Indica que uma carga formal ou parcial não pôde ser interpretada.

    Pode ocorrer quando:
        - a carga possui formato desconhecido;
        - o valor não é numérico;
        - a carga é ambígua;
        - o modelo químico não oferece informação suficiente.
    """


class AmbiguousChargeError(ChargedGroupError):
    """
    Indica que a polaridade de um átomo ou grupo não pôde ser determinada
    de forma inequívoca.
    """


class SaltBridgeGeometryError(SaltBridgeError):
    """
    Indica falha em um cálculo geométrico de ponte salina.

    Exemplos:
        - coordenadas inválidas;
        - vetores com dimensão incorreta;
        - valores não finitos;
        - centro geométrico impossível de calcular.
    """


class MissingCoordinatesError(SaltBridgeGeometryError):
    """
    Indica ausência de coordenadas utilizáveis em um átomo ou grupo.
    """


class DegenerateGeometryError(SaltBridgeGeometryError):
    """
    Indica uma geometria degenerada que impede a avaliação da interação.

    Exemplos:
        - grupo sem átomos válidos;
        - centro geométrico indefinido;
        - coordenadas coincidentes em uma operação que exige separação.
    """


class SaltBridgeDetectionError(SaltBridgeError):
    """
    Indica falha durante a busca ou detecção central de pontes salinas.
    """


class InvalidInteractionError(SaltBridgeDetectionError, ValueError):
    """
    Indica que um resultado de interação é internamente inconsistente.

    Exemplos:
        - dois grupos com a mesma polaridade;
        - distância negativa;
        - cátion ou ânion ausente;
        - classificação incompatível com a geometria.
    """


class SaltBridgeScoringError(SaltBridgeError):
    """
    Indica falha durante a classificação ou o cálculo do score.
    """


class SaltBridgeIntegrationError(SaltBridgeError):
    """
    Indica falha de integração com objetos externos ao núcleo geométrico.

    Exemplos:
        - DockModel incompatível;
        - atributo de destino ausente;
        - estrutura molecular não reconhecida;
        - atualização de resultados impossível.
    """


class DockModelSaltBridgeError(SaltBridgeIntegrationError):
    """
    Indica especificamente uma falha ao anexar ou recuperar resultados
    de pontes salinas em um DockModel.
    """


class SaltBridgeSerializationError(SaltBridgeError):
    """
    Indica falha na conversão ou serialização dos resultados.

    Exemplos:
        - objeto não serializável;
        - referência circular;
        - campo obrigatório ausente;
        - formato de exportação não suportado.
    """


class ChimeraXSaltBridgeError(SaltBridgeIntegrationError):
    """
    Exceção-base para erros da camada opcional de compatibilidade com ChimeraX.
    """


class ChimeraXUnavailableError(ChimeraXSaltBridgeError, RuntimeError):
    """
    Indica que uma função dependente do ChimeraX foi chamada em um ambiente
    no qual o ChimeraX não está disponível.
    """


class SaltBridgeSelfTestError(SaltBridgeError, AssertionError):
    """
    Indica falha controlada nos self-tests do módulo.

    Essa exceção permite diferenciar uma asserção dos testes internos de
    outras falhas de execução ocorridas durante a análise.
    """


# =============================================================================
# 4. FUNDAMENTAL DATACLASSES
# =============================================================================


@dataclass(slots=True)
class ChargedAtom:
    """
    Lightweight representation of an atom involved in a charged group.

    The original atom object is preserved by reference. This avoids duplicating
    molecular data and allows later integration with ChimeraX, DockModel, or
    other molecular representations.

    Attributes
    ----------
    atom
        Reference to the original atom object.
    coordinate
        Cartesian coordinate in angstroms.
    element
        Normalized chemical element symbol.
    name
        Atom name as provided by the molecular structure.
    residue
        Reference to the parent residue, when available.
    formal_charge
        Formal atomic charge, when explicitly available.
    partial_charge
        Partial atomic charge, when explicitly available.
    polarity
        Charge polarity: ``"positive"``, ``"negative"``, or ``"neutral"``.
    source
        Method used to identify the charge.
    metadata
        Small optional dictionary containing additional information.
    """

    atom: AtomLike
    coordinate: Optional[Coordinate] = None
    element: str = ""
    name: str = ""
    residue: Optional[ResidueLike] = None
    formal_charge: Optional[float] = None
    partial_charge: Optional[float] = None
    polarity: str = "neutral"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the basic charged-atom attributes."""

        self.element = str(self.element or "").strip().upper()
        self.name = str(self.name or "").strip()
        self.polarity = str(self.polarity or "neutral").strip().lower()
        self.source = str(self.source or "unknown").strip().lower()

        if self.polarity not in {"positive", "negative", "neutral"}:
            raise InvalidChargedGroupError(
                f"Unsupported atom polarity: {self.polarity!r}."
            )

        if self.coordinate is not None:
            if len(self.coordinate) != 3:
                raise MissingCoordinatesError(
                    "Atom coordinates must contain exactly three components."
                )

            normalized_coordinate = tuple(
                float(value) for value in self.coordinate
            )

            if not all(math.isfinite(value) for value in normalized_coordinate):
                raise MissingCoordinatesError(
                    "Atom coordinates must contain only finite numeric values."
                )

            self.coordinate = normalized_coordinate

        if self.formal_charge is not None:
            self.formal_charge = float(self.formal_charge)

            if not math.isfinite(self.formal_charge):
                raise UnsupportedChargeError(
                    "Formal charge must be a finite numeric value."
                )

        if self.partial_charge is not None:
            self.partial_charge = float(self.partial_charge)

            if not math.isfinite(self.partial_charge):
                raise UnsupportedChargeError(
                    "Partial charge must be a finite numeric value."
                )

    @property
    def has_coordinates(self) -> bool:
        """Return whether the atom has valid Cartesian coordinates."""

        return self.coordinate is not None

    @property
    def effective_charge(self) -> Optional[float]:
        """
        Return the best available numeric charge.

        Formal charge is preferred because it directly represents the
        chemically assigned ionic state. Partial charge is used only when
        formal charge is unavailable.
        """

        if self.formal_charge is not None:
            return self.formal_charge

        return self.partial_charge

    @property
    def is_positive(self) -> bool:
        """Return whether the atom is classified as positively charged."""

        return self.polarity == "positive"

    @property
    def is_negative(self) -> bool:
        """Return whether the atom is classified as negatively charged."""

        return self.polarity == "negative"


@dataclass(slots=True)
class ChargedGroup:
    """
    Representation of a chemically meaningful charged atomic group.

    A charged group may contain a single charged atom, such as the lysine NZ
    atom, or multiple atoms sharing a delocalized charge, such as the
    guanidinium group of arginine or the carboxylate group of aspartate.

    Attributes
    ----------
    atoms
        Tuple containing the charged atoms that define the group.
    polarity
        Group polarity: ``"positive"`` or ``"negative"``.
    group_type
        Chemical classification of the group.
    center
        Representative Cartesian center of the charged group.
    net_charge
        Estimated or explicit net charge of the group.
    residue
        Reference to the parent residue, when available.
    representative_atom
        Atom selected as the primary representative of the group.
    source
        Recognition strategy that produced the group.
    confidence
        Recognition confidence between 0.0 and 1.0.
    group_id
        Stable optional identifier used during deduplication and export.
    metadata
        Small optional dictionary containing additional information.
    """

    atoms: Tuple[ChargedAtom, ...]
    polarity: str
    group_type: str = "unknown"
    center: Optional[Coordinate] = None
    net_charge: Optional[float] = None
    residue: Optional[ResidueLike] = None
    representative_atom: Optional[ChargedAtom] = None
    source: str = "unknown"
    confidence: float = 1.0
    group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the charged-group attributes."""

        self.atoms = tuple(self.atoms)
        self.polarity = str(self.polarity or "").strip().lower()
        self.group_type = str(self.group_type or "unknown").strip().lower()
        self.source = str(self.source or "unknown").strip().lower()

        if not self.atoms:
            raise InvalidChargedGroupError(
                "A charged group must contain at least one atom."
            )

        if self.polarity not in {"positive", "negative"}:
            raise InvalidChargedGroupError(
                "Charged-group polarity must be either "
                "'positive' or 'negative'."
            )

        if self.center is not None:
            if len(self.center) != 3:
                raise MissingCoordinatesError(
                    "Group center coordinates must contain exactly "
                    "three components."
                )

            normalized_center = tuple(float(value) for value in self.center)

            if not all(math.isfinite(value) for value in normalized_center):
                raise MissingCoordinatesError(
                    "Group center coordinates must contain only finite values."
                )

            self.center = normalized_center

        if self.net_charge is not None:
            self.net_charge = float(self.net_charge)

            if not math.isfinite(self.net_charge):
                raise UnsupportedChargeError(
                    "Group net charge must be a finite numeric value."
                )

            if self.polarity == "positive" and self.net_charge < 0.0:
                raise InvalidChargedGroupError(
                    "A positive group cannot have a negative net charge."
                )

            if self.polarity == "negative" and self.net_charge > 0.0:
                raise InvalidChargedGroupError(
                    "A negative group cannot have a positive net charge."
                )

        self.confidence = float(self.confidence)

        if not math.isfinite(self.confidence):
            raise InvalidChargedGroupError(
                "Group confidence must be a finite numeric value."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidChargedGroupError(
                "Group confidence must be between 0.0 and 1.0."
            )

        if self.representative_atom is None:
            self.representative_atom = self.atoms[0]

        elif self.representative_atom not in self.atoms:
            raise InvalidChargedGroupError(
                "The representative atom must belong to the charged group."
            )

    @property
    def atom_count(self) -> int:
        """Return the number of atoms defining the charged group."""

        return len(self.atoms)

    @property
    def is_positive(self) -> bool:
        """Return whether the group is positively charged."""

        return self.polarity == "positive"

    @property
    def is_negative(self) -> bool:
        """Return whether the group is negatively charged."""

        return self.polarity == "negative"

    @property
    def has_center(self) -> bool:
        """Return whether the group has a representative center."""

        return self.center is not None

    @property
    def original_atoms(self) -> Tuple[AtomLike, ...]:
        """Return references to the original molecular atom objects."""

        return tuple(charged_atom.atom for charged_atom in self.atoms)

    @property
    def coordinates(self) -> Tuple[Coordinate, ...]:
        """Return all available charged-atom coordinates."""

        return tuple(
            charged_atom.coordinate
            for charged_atom in self.atoms
            if charged_atom.coordinate is not None
        )


@dataclass(slots=True)
class SaltBridgeGeometry:
    """
    Geometric description of a candidate salt bridge.

    The object stores only the geometric values required for classification,
    scoring, reporting, and debugging. It does not store the complete
    atom-by-atom distance matrix.

    Attributes
    ----------
    center_distance
        Distance between the representative centers of the charged groups.
    minimum_atom_distance
        Shortest distance between atoms belonging to opposite groups.
    maximum_atom_distance
        Largest retained intergroup atomic distance, when calculated.
    mean_atom_distance
        Mean retained intergroup atomic distance, when calculated.
    contact_count
        Number of atom pairs satisfying the atomic contact cutoff.
    closest_positive_atom
        Positive atom involved in the shortest contact.
    closest_negative_atom
        Negative atom involved in the shortest contact.
    valid
        Whether the geometry satisfies the central detection criteria.
    rejection_reason
        Explanation for rejected geometries.
    """

    center_distance: float
    minimum_atom_distance: float
    maximum_atom_distance: Optional[float] = None
    mean_atom_distance: Optional[float] = None
    contact_count: int = 0
    closest_positive_atom: Optional[ChargedAtom] = None
    closest_negative_atom: Optional[ChargedAtom] = None
    valid: bool = True
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the geometric measurements."""

        self.center_distance = float(self.center_distance)
        self.minimum_atom_distance = float(self.minimum_atom_distance)
        self.contact_count = int(self.contact_count)

        geometric_values = {
            "center_distance": self.center_distance,
            "minimum_atom_distance": self.minimum_atom_distance,
        }

        if self.maximum_atom_distance is not None:
            self.maximum_atom_distance = float(self.maximum_atom_distance)
            geometric_values["maximum_atom_distance"] = (
                self.maximum_atom_distance
            )

        if self.mean_atom_distance is not None:
            self.mean_atom_distance = float(self.mean_atom_distance)
            geometric_values["mean_atom_distance"] = self.mean_atom_distance

        for name, value in geometric_values.items():
            if not math.isfinite(value):
                raise SaltBridgeGeometryError(
                    f"{name} must be a finite numeric value."
                )

            if value < 0.0:
                raise SaltBridgeGeometryError(
                    f"{name} cannot be negative."
                )

        if self.contact_count < 0:
            raise SaltBridgeGeometryError(
                "The contact count cannot be negative."
            )

        if (
            self.maximum_atom_distance is not None
            and self.maximum_atom_distance < self.minimum_atom_distance
        ):
            raise SaltBridgeGeometryError(
                "The maximum atom distance cannot be smaller than "
                "the minimum atom distance."
            )

        if (
            self.mean_atom_distance is not None
            and self.maximum_atom_distance is not None
            and not (
                self.minimum_atom_distance
                <= self.mean_atom_distance
                <= self.maximum_atom_distance
            )
        ):
            raise SaltBridgeGeometryError(
                "The mean atom distance must lie between the minimum "
                "and maximum atom distances."
            )

        if self.valid:
            self.rejection_reason = None

    @property
    def closest_atom_pair(
        self,
    ) -> Tuple[Optional[ChargedAtom], Optional[ChargedAtom]]:
        """Return the closest positive-negative atom pair."""

        return (
            self.closest_positive_atom,
            self.closest_negative_atom,
        )


@dataclass(slots=True)
class SaltBridgeInteraction:
    """
    Complete representation of one detected salt bridge.

    The interaction stores references to the participating charged groups and
    a compact geometric result. Classification and scoring fields may be
    populated during the detection step or updated later by Section 10.

    Attributes
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    geometry
        Geometric measurements associated with the interaction.
    interaction_type
        Interaction classification.
    strength
        Qualitative strength class.
    score
        Numeric interaction score.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    interaction_id
        Optional stable identifier used in deduplication and serialization.
    metadata
        Small optional dictionary containing additional information.
    """

    cation: ChargedGroup
    anion: ChargedGroup
    geometry: SaltBridgeGeometry
    interaction_type: str = SALT_BRIDGE
    strength: str = STRENGTH_WEAK
    score: float = 0.0
    pose_id: Optional[Union[str, int]] = None
    model_id: Optional[Union[str, int]] = None
    interaction_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the salt-bridge interaction."""

        self.interaction_type = str(
            self.interaction_type or SALT_BRIDGE
        ).strip().lower()

        self.strength = str(
            self.strength or STRENGTH_WEAK
        ).strip().lower()

        self.score = float(self.score)

        if not self.cation.is_positive:
            raise InvalidInteractionError(
                "The cation group must have positive polarity."
            )

        if not self.anion.is_negative:
            raise InvalidInteractionError(
                "The anion group must have negative polarity."
            )

        if self.strength not in STRENGTH_ORDER:
            raise InvalidInteractionError(
                f"Unsupported salt-bridge strength: {self.strength!r}."
            )

        if not math.isfinite(self.score):
            raise SaltBridgeScoringError(
                "The interaction score must be a finite numeric value."
            )

        if self.score < 0.0:
            raise SaltBridgeScoringError(
                "The interaction score cannot be negative."
            )

    @property
    def distance(self) -> float:
        """
        Return the primary interaction distance.

        The minimum atom-to-atom distance is used as the default interaction
        distance because it directly represents the closest electrostatic
        contact.
        """

        return self.geometry.minimum_atom_distance

    @property
    def center_distance(self) -> float:
        """Return the distance between the charged-group centers."""

        return self.geometry.center_distance

    @property
    def is_valid(self) -> bool:
        """Return whether the interaction passed the geometric criteria."""

        return self.geometry.valid

    @property
    def groups(self) -> Tuple[ChargedGroup, ChargedGroup]:
        """Return the cation and anion as an ordered pair."""

        return self.cation, self.anion

    @property
    def residues(
        self,
    ) -> Tuple[Optional[ResidueLike], Optional[ResidueLike]]:
        """Return the residues associated with the cation and anion."""

        return self.cation.residue, self.anion.residue


@dataclass(slots=True)
class SaltBridgeResult:
    """
    Container holding the complete result of a salt-bridge analysis.

    This object centralizes recognized groups, detected interactions,
    warnings, statistics, and analysis metadata. The statistics dictionary
    may remain empty until Section 13 is applied.

    Attributes
    ----------
    interactions
        Detected and retained salt-bridge interactions.
    cationic_groups
        Recognized positively charged groups.
    anionic_groups
        Recognized negatively charged groups.
    statistics
        Calculated summary statistics.
    warnings
        Non-fatal messages produced during analysis.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    metadata
        Additional compact analysis information.
    """

    interactions: List[SaltBridgeInteraction] = field(default_factory=list)
    cationic_groups: List[ChargedGroup] = field(default_factory=list)
    anionic_groups: List[ChargedGroup] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    pose_id: Optional[Union[str, int]] = None
    model_id: Optional[Union[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the polarity of the recognized charged groups."""

        for group in self.cationic_groups:
            if not group.is_positive:
                raise InvalidChargedGroupError(
                    "All cationic_groups entries must be positively charged."
                )

        for group in self.anionic_groups:
            if not group.is_negative:
                raise InvalidChargedGroupError(
                    "All anionic_groups entries must be negatively charged."
                )

    def __len__(self) -> int:
        """Return the number of detected interactions."""

        return len(self.interactions)

    def __iter__(self) -> Iterator[SaltBridgeInteraction]:
        """Iterate over detected salt-bridge interactions."""

        return iter(self.interactions)

    def __bool__(self) -> bool:
        """Return whether at least one interaction was detected."""

        return bool(self.interactions)

    @property
    def interaction_count(self) -> int:
        """Return the number of detected interactions."""

        return len(self.interactions)

    @property
    def cation_count(self) -> int:
        """Return the number of recognized cationic groups."""

        return len(self.cationic_groups)

    @property
    def anion_count(self) -> int:
        """Return the number of recognized anionic groups."""

        return len(self.anionic_groups)

    @property
    def total_score(self) -> float:
        """Return the sum of all interaction scores."""

        return float(
            sum(interaction.score for interaction in self.interactions)
        )

    @property
    def valid_interactions(self) -> Tuple[SaltBridgeInteraction, ...]:
        """Return only interactions with valid geometry."""

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.is_valid
        )

    def add_interaction(
        self,
        interaction: SaltBridgeInteraction,
    ) -> None:
        """Append one validated salt-bridge interaction."""

        if not isinstance(interaction, SaltBridgeInteraction):
            raise InvalidInteractionError(
                "Only SaltBridgeInteraction instances can be added."
            )

        self.interactions.append(interaction)

    def add_warning(self, message: str) -> None:
        """Add a non-empty warning message without duplicating it."""

        normalized_message = str(message or "").strip()

        if normalized_message and normalized_message not in self.warnings:
            self.warnings.append(normalized_message)



# =============================================================================
# 5. CONFIGURATION
# =============================================================================


@dataclass(slots=True)
class SaltBridgeConfig:
    """
    Configuration object controlling salt-bridge recognition and detection.

    The configuration separates chemical-recognition parameters from geometric,
    scoring, deduplication, integration, and serialization options. All values
    are validated during initialization so that later sections can assume a
    consistent configuration state.

    Attributes
    ----------
    distance_cutoff
        Maximum atom-to-atom distance, in angstroms, for accepting a salt
        bridge.
    center_distance_cutoff
        Maximum allowed distance between charged-group centers.
    strong_distance_cutoff
        Maximum minimum atom distance classified as a strong interaction.
    moderate_distance_cutoff
        Maximum minimum atom distance classified as a moderate interaction.
    minimum_contact_distance
        Optional lower distance bound used to reject geometrically implausible
        atomic overlaps.
    atomic_contact_cutoff
        Maximum distance used when counting individual atom-to-atom contacts.
    minimum_contact_count
        Minimum number of atomic contacts required for an accepted interaction.
    use_center_distance
        Whether the group-center distance must satisfy its cutoff.
    use_minimum_atom_distance
        Whether the shortest atom-to-atom distance must satisfy its cutoff.
    calculate_all_contact_distances
        Whether mean and maximum retained atomic distances should be calculated.
    include_protein_groups
        Whether canonical protein charged groups should be recognized.
    include_ligand_groups
        Whether charged groups from non-protein residues should be recognized.
    include_nucleic_acid_groups
        Whether charged groups from nucleic acids should be recognized.
    recognize_canonical_residues
        Whether standard residue-name and atom-name rules should be applied.
    recognize_formal_charges
        Whether explicit formal charges should be used.
    recognize_partial_charges
        Whether partial charges may be used when formal charges are unavailable.
    infer_charge_from_chemistry
        Whether charge may be inferred from atom names, residue identities,
        and local chemical patterns.
    partial_charge_positive_threshold
        Minimum partial charge used to classify an atom as positive.
    partial_charge_negative_threshold
        Maximum partial charge used to classify an atom as negative.
    minimum_group_charge
        Minimum absolute estimated group charge required for recognition.
    allow_histidine_cations
        Whether positively protonated histidine variants should be recognized.
    allow_terminal_groups
        Whether charged protein termini should be recognized.
    allow_ambiguous_groups
        Whether groups with uncertain charge assignments may be retained.
    minimum_recognition_confidence
        Minimum recognition confidence accepted for a charged group.
    deduplicate_groups
        Whether duplicate charged groups should be removed.
    deduplicate_interactions
        Whether duplicate interactions should be removed.
    deduplication_distance_tolerance
        Distance tolerance used when comparing potentially duplicate results.
    scoring_enabled
        Whether interaction scores should be calculated.
    strong_score
        Base score assigned to strong interactions.
    moderate_score
        Base score assigned to moderate interactions.
    weak_score
        Base score assigned to weak interactions.
    contact_count_bonus
        Additional score added for each contact beyond the minimum requirement.
    maximum_contact_bonus
        Maximum total bonus derived from multiple atomic contacts.
    confidence_weighting
        Whether recognition confidence should influence the interaction score.
    charge_weighting
        Whether estimated group charges should influence the interaction score.
    preserve_invalid_candidates
        Whether rejected candidate geometries should be preserved for debugging.
    compact_results
        Whether optional heavy metadata should be omitted from results.
    preserve_existing_results
        Whether DockModel integration should preserve previous results.
    update_dockmodel_statistics
        Whether DockModel statistics should be updated after analysis.
    update_dockmodel_score
        Whether the DockModel total score should include salt-bridge scoring.
    strict
        Whether uncertain or malformed inputs should raise exceptions instead
        of producing warnings and skipping invalid entries.
    """

    # -------------------------------------------------------------------------
    # Geometric criteria
    # -------------------------------------------------------------------------

    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF
    center_distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF
    strong_distance_cutoff: float = DEFAULT_STRONG_CUTOFF
    moderate_distance_cutoff: float = 3.6
    minimum_contact_distance: float = 1.5
    atomic_contact_cutoff: float = DEFAULT_DISTANCE_CUTOFF
    minimum_contact_count: int = 1

    use_center_distance: bool = True
    use_minimum_atom_distance: bool = True
    calculate_all_contact_distances: bool = False

    # -------------------------------------------------------------------------
    # Recognition options
    # -------------------------------------------------------------------------

    include_protein_groups: bool = True
    include_ligand_groups: bool = True
    include_nucleic_acid_groups: bool = True

    recognize_canonical_residues: bool = True
    recognize_formal_charges: bool = True
    recognize_partial_charges: bool = True
    infer_charge_from_chemistry: bool = True

    partial_charge_positive_threshold: float = 0.30
    partial_charge_negative_threshold: float = -0.30
    minimum_group_charge: float = 0.50

    allow_histidine_cations: bool = True
    allow_terminal_groups: bool = True
    allow_ambiguous_groups: bool = False

    minimum_recognition_confidence: float = 0.50

    # -------------------------------------------------------------------------
    # Deduplication options
    # -------------------------------------------------------------------------

    deduplicate_groups: bool = True
    deduplicate_interactions: bool = True
    deduplication_distance_tolerance: float = 0.05

    # -------------------------------------------------------------------------
    # Scoring options
    # -------------------------------------------------------------------------

    scoring_enabled: bool = True

    strong_score: float = DEFAULT_SCORE_STRONG
    moderate_score: float = DEFAULT_SCORE_MODERATE
    weak_score: float = DEFAULT_SCORE_WEAK

    contact_count_bonus: float = 0.05
    maximum_contact_bonus: float = 0.25

    confidence_weighting: bool = True
    charge_weighting: bool = False

    # -------------------------------------------------------------------------
    # Result and integration options
    # -------------------------------------------------------------------------

    preserve_invalid_candidates: bool = False
    compact_results: bool = False

    preserve_existing_results: bool = True
    update_dockmodel_statistics: bool = True
    update_dockmodel_score: bool = True

    # -------------------------------------------------------------------------
    # Error-handling behavior
    # -------------------------------------------------------------------------

    strict: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate all configuration parameters."""

        self._normalize_numeric_values()
        self._validate_distance_parameters()
        self._validate_recognition_parameters()
        self._validate_scoring_parameters()
        self._validate_deduplication_parameters()

    def _normalize_numeric_values(self) -> None:
        """Convert numeric configuration values to stable built-in types."""

        float_fields = (
            "distance_cutoff",
            "center_distance_cutoff",
            "strong_distance_cutoff",
            "moderate_distance_cutoff",
            "minimum_contact_distance",
            "atomic_contact_cutoff",
            "partial_charge_positive_threshold",
            "partial_charge_negative_threshold",
            "minimum_group_charge",
            "minimum_recognition_confidence",
            "deduplication_distance_tolerance",
            "strong_score",
            "moderate_score",
            "weak_score",
            "contact_count_bonus",
            "maximum_contact_bonus",
        )

        for field_name in float_fields:
            value = float(getattr(self, field_name))

            if not math.isfinite(value):
                raise SaltBridgeConfigurationError(
                    f"{field_name} must be a finite numeric value."
                )

            setattr(self, field_name, value)

        self.minimum_contact_count = int(self.minimum_contact_count)

    def _validate_distance_parameters(self) -> None:
        """Validate geometric cutoffs and their ordering."""

        positive_distance_fields = (
            "distance_cutoff",
            "center_distance_cutoff",
            "strong_distance_cutoff",
            "moderate_distance_cutoff",
            "minimum_contact_distance",
            "atomic_contact_cutoff",
        )

        for field_name in positive_distance_fields:
            value = getattr(self, field_name)

            if value <= 0.0:
                raise SaltBridgeConfigurationError(
                    f"{field_name} must be greater than zero."
                )

        if self.minimum_contact_count < 1:
            raise SaltBridgeConfigurationError(
                "minimum_contact_count must be at least 1."
            )

        if self.minimum_contact_distance >= self.distance_cutoff:
            raise SaltBridgeConfigurationError(
                "minimum_contact_distance must be smaller than "
                "distance_cutoff."
            )

        if self.strong_distance_cutoff > self.moderate_distance_cutoff:
            raise SaltBridgeConfigurationError(
                "strong_distance_cutoff cannot exceed "
                "moderate_distance_cutoff."
            )

        if self.moderate_distance_cutoff > self.distance_cutoff:
            raise SaltBridgeConfigurationError(
                "moderate_distance_cutoff cannot exceed distance_cutoff."
            )

        if self.atomic_contact_cutoff > self.distance_cutoff:
            raise SaltBridgeConfigurationError(
                "atomic_contact_cutoff cannot exceed distance_cutoff."
            )

        if not (
            self.use_center_distance
            or self.use_minimum_atom_distance
        ):
            raise SaltBridgeConfigurationError(
                "At least one geometric distance criterion must be enabled."
            )

    def _validate_recognition_parameters(self) -> None:
        """Validate charge-recognition thresholds and related options."""

        if self.partial_charge_positive_threshold <= 0.0:
            raise SaltBridgeConfigurationError(
                "partial_charge_positive_threshold must be greater than zero."
            )

        if self.partial_charge_negative_threshold >= 0.0:
            raise SaltBridgeConfigurationError(
                "partial_charge_negative_threshold must be smaller than zero."
            )

        if self.minimum_group_charge < 0.0:
            raise SaltBridgeConfigurationError(
                "minimum_group_charge cannot be negative."
            )

        if not 0.0 <= self.minimum_recognition_confidence <= 1.0:
            raise SaltBridgeConfigurationError(
                "minimum_recognition_confidence must be between 0.0 and 1.0."
            )

        if not (
            self.recognize_canonical_residues
            or self.recognize_formal_charges
            or self.recognize_partial_charges
            or self.infer_charge_from_chemistry
        ):
            raise SaltBridgeConfigurationError(
                "At least one charge-recognition strategy must be enabled."
            )

        if not (
            self.include_protein_groups
            or self.include_ligand_groups
            or self.include_nucleic_acid_groups
        ):
            raise SaltBridgeConfigurationError(
                "At least one molecular group category must be enabled."
            )

    def _validate_scoring_parameters(self) -> None:
        """Validate scoring values and their expected ordering."""

        score_fields = (
            "strong_score",
            "moderate_score",
            "weak_score",
            "contact_count_bonus",
            "maximum_contact_bonus",
        )

        for field_name in score_fields:
            if getattr(self, field_name) < 0.0:
                raise SaltBridgeConfigurationError(
                    f"{field_name} cannot be negative."
                )

        if self.strong_score < self.moderate_score:
            raise SaltBridgeConfigurationError(
                "strong_score cannot be smaller than moderate_score."
            )

        if self.moderate_score < self.weak_score:
            raise SaltBridgeConfigurationError(
                "moderate_score cannot be smaller than weak_score."
            )

        if self.contact_count_bonus > self.maximum_contact_bonus:
            raise SaltBridgeConfigurationError(
                "contact_count_bonus cannot exceed maximum_contact_bonus."
            )

    def _validate_deduplication_parameters(self) -> None:
        """Validate group and interaction deduplication settings."""

        if self.deduplication_distance_tolerance < 0.0:
            raise SaltBridgeConfigurationError(
                "deduplication_distance_tolerance cannot be negative."
            )

    def copy_with(self, **changes: Any) -> "SaltBridgeConfig":
        """
        Return a validated configuration containing selected field changes.

        This method avoids modifying the current configuration in place and is
        useful when a temporary analysis requires different cutoffs or scoring
        behavior.

        Parameters
        ----------
        **changes
            Configuration fields and their replacement values.

        Returns
        -------
        SaltBridgeConfig
            New validated configuration instance.
        """

        valid_fields = {
            field_name
            for field_name in self.__dataclass_fields__
        }

        unknown_fields = set(changes) - valid_fields

        if unknown_fields:
            formatted_fields = ", ".join(sorted(unknown_fields))

            raise SaltBridgeConfigurationError(
                f"Unknown configuration field or fields: {formatted_fields}."
            )

        current_values = {
            field_name: getattr(self, field_name)
            for field_name in valid_fields
        }

        current_values.update(changes)

        return type(self)(**current_values)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the configuration as a plain dictionary.

        The returned dictionary contains only built-in scalar values and can
        therefore be safely reused by later serialization functions.
        """

        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


DEFAULT_SALT_BRIDGE_CONFIG = SaltBridgeConfig()


# =============================================================================
# 6. GENERAL UTILITIES
# =============================================================================


_MISSING = object()


def normalize_text(
    value: Any,
    *,
    default: str = "",
    uppercase: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Convert a value to a normalized stripped string.

    Parameters
    ----------
    value
        Value to normalize.
    default
        Value returned when the input is ``None`` or produces an empty string.
    uppercase
        Whether the result should be converted to uppercase.
    lowercase
        Whether the result should be converted to lowercase.

    Returns
    -------
    str
        Normalized string.

    Raises
    ------
    ValueError
        If both uppercase and lowercase conversion are requested.
    """

    if uppercase and lowercase:
        raise ValueError(
            "uppercase and lowercase cannot both be enabled."
        )

    if value is None:
        text = str(default)
    else:
        text = str(value).strip()

        if not text:
            text = str(default)

    if uppercase:
        return text.upper()

    if lowercase:
        return text.lower()

    return text


def safe_float(
    value: Any,
    *,
    default: Optional[float] = None,
    finite_only: bool = True,
) -> Optional[float]:
    """
    Convert a value to float without propagating ordinary conversion errors.

    Parameters
    ----------
    value
        Value to convert.
    default
        Value returned when conversion is not possible.
    finite_only
        Whether infinite and NaN values should be rejected.

    Returns
    -------
    Optional[float]
        Converted value or the provided default.
    """

    if value is None:
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if finite_only and not math.isfinite(converted):
        return default

    return converted


def safe_int(
    value: Any,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Convert a value to int without propagating ordinary conversion errors.

    Parameters
    ----------
    value
        Value to convert.
    default
        Value returned when conversion is not possible.

    Returns
    -------
    Optional[int]
        Converted integer or the provided default.
    """

    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_getattr(
    obj: Any,
    names: Union[str, Sequence[str]],
    *,
    default: Any = None,
    call: bool = False,
) -> Any:
    """
    Return the first accessible attribute from a sequence of candidate names.

    Parameters
    ----------
    obj
        Object from which attributes should be retrieved.
    names
        Single attribute name or ordered sequence of candidate names.
    default
        Value returned when no candidate attribute can be accessed.
    call
        Whether a callable attribute should be invoked without arguments.

    Returns
    -------
    Any
        First successfully retrieved value or the provided default.

    Notes
    -----
    Attribute access failures are intentionally ignored because molecular
    objects from different libraries may expose partially compatible APIs.
    """

    if obj is None:
        return default

    if isinstance(names, str):
        candidate_names = (names,)
    else:
        candidate_names = tuple(names)

    for name in candidate_names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if call and callable(value):
            try:
                value = value()
            except Exception:
                continue

        if value is not None:
            return value

    return default


def safe_mapping_get(
    mapping: Any,
    keys: Union[str, Sequence[str]],
    *,
    default: Any = None,
) -> Any:
    """
    Return the first available value from a mapping using candidate keys.

    Parameters
    ----------
    mapping
        Mapping-like object.
    keys
        Single key or ordered sequence of candidate keys.
    default
        Value returned when no key is available.

    Returns
    -------
    Any
        Retrieved value or the provided default.
    """

    if mapping is None:
        return default

    if isinstance(keys, str):
        candidate_keys = (keys,)
    else:
        candidate_keys = tuple(keys)

    for key in candidate_keys:
        try:
            if key in mapping:
                value = mapping[key]

                if value is not None:
                    return value
        except Exception:
            continue

    return default


def get_value(
    obj: Any,
    names: Union[str, Sequence[str]],
    *,
    default: Any = None,
    call: bool = False,
) -> Any:
    """
    Retrieve a value from either attributes or mapping keys.

    Attribute access is attempted first, followed by mapping lookup.

    Parameters
    ----------
    obj
        Source object or mapping.
    names
        Candidate attribute or key names.
    default
        Value returned when no candidate is available.
    call
        Whether callable attributes should be invoked.

    Returns
    -------
    Any
        Retrieved value or the provided default.
    """

    value = safe_getattr(
        obj,
        names,
        default=_MISSING,
        call=call,
    )

    if value is not _MISSING:
        return value

    return safe_mapping_get(
        obj,
        names,
        default=default,
    )


def normalize_element(value: Any) -> str:
    """
    Normalize an element representation to an uppercase chemical symbol.

    Parameters
    ----------
    value
        Element name, symbol, atomic object, or atomic number.

    Returns
    -------
    str
        Normalized element symbol or an empty string when unavailable.
    """

    if value is None:
        return ""

    if isinstance(value, int):
        atomic_number_map = {
            1: "H",
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
            26: "FE",
            30: "ZN",
            35: "BR",
            53: "I",
        }

        return atomic_number_map.get(value, "")

    nested_name = safe_getattr(
        value,
        ("name", "symbol"),
        default=None,
    )

    if nested_name is not None and nested_name is not value:
        value = nested_name

    text = normalize_text(value, uppercase=True)

    if not text:
        return ""

    if text.isdigit():
        return normalize_element(int(text))

    aliases = {
        "HYDROGEN": "H",
        "CARBON": "C",
        "NITROGEN": "N",
        "OXYGEN": "O",
        "PHOSPHORUS": "P",
        "SULFUR": "S",
        "SULPHUR": "S",
        "FLUORINE": "F",
        "CHLORINE": "CL",
        "BROMINE": "BR",
        "IODINE": "I",
        "SODIUM": "NA",
        "POTASSIUM": "K",
        "CALCIUM": "CA",
        "MAGNESIUM": "MG",
        "IRON": "FE",
        "ZINC": "ZN",
    }

    if text in aliases:
        return aliases[text]

    letters = "".join(character for character in text if character.isalpha())

    if not letters:
        return ""

    if len(letters) == 1:
        return letters

    return letters[:2]


def infer_element_from_atom_name(atom_name: Any) -> str:
    """
    Infer an element symbol from a molecular atom name.

    Parameters
    ----------
    atom_name
        Atom name such as ``"NZ"``, ``"OD1"``, or ``"CL1"``.

    Returns
    -------
    str
        Inferred uppercase element symbol.

    Notes
    -----
    This function performs only syntactic inference. Chemical validation
    remains the responsibility of the recognition section.
    """

    text = normalize_text(atom_name, uppercase=True)

    if not text:
        return ""

    text = text.lstrip("0123456789")

    if not text:
        return ""

    two_letter_elements = {
        "BR",
        "CA",
        "CL",
        "FE",
        "MG",
        "NA",
        "ZN",
    }

    if len(text) >= 2 and text[:2] in two_letter_elements:
        return text[:2]

    return text[0]


def get_atom_name(atom: AtomLike) -> str:
    """
    Return a normalized atom name.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    str
        Atom name with surrounding whitespace removed.
    """

    value = get_value(
        atom,
        ("name", "atom_name", "atomName"),
        default="",
    )

    return normalize_text(value)


def get_atom_element(atom: AtomLike) -> str:
    """
    Return the normalized chemical element of an atom.

    Explicit element information is preferred. When unavailable, the element
    is inferred from the atom name.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    str
        Uppercase element symbol.
    """

    value = get_value(
        atom,
        (
            "element",
            "element_name",
            "element_symbol",
            "atomic_number",
        ),
        default=None,
    )

    element = normalize_element(value)

    if element:
        return element

    return infer_element_from_atom_name(get_atom_name(atom))


def get_atom_residue(atom: AtomLike) -> Optional[ResidueLike]:
    """
    Return the parent residue of an atom when available.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Optional[ResidueLike]
        Parent residue reference or ``None``.
    """

    return get_value(
        atom,
        ("residue", "parent_residue", "res"),
        default=None,
    )


def get_residue_name(residue: Optional[ResidueLike]) -> str:
    """
    Return a normalized uppercase residue name.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    str
        Uppercase residue name.
    """

    value = get_value(
        residue,
        ("name", "resname", "residue_name", "type"),
        default="",
    )

    return normalize_text(value, uppercase=True)


def get_residue_number(
    residue: Optional[ResidueLike],
) -> Optional[Union[int, str]]:
    """
    Return the residue sequence number or identifier.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    Optional[Union[int, str]]
        Residue number, identifier, or ``None``.
    """

    value = get_value(
        residue,
        (
            "number",
            "resid",
            "residue_number",
            "sequence_number",
            "id",
        ),
        default=None,
    )

    if value is None:
        return None

    integer_value = safe_int(value)

    if integer_value is not None:
        return integer_value

    normalized_value = normalize_text(value)

    return normalized_value or None


def get_chain_id(residue: Optional[ResidueLike]) -> str:
    """
    Return the normalized chain identifier associated with a residue.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    str
        Chain identifier or an empty string.
    """

    chain_value = get_value(
        residue,
        (
            "chain_id",
            "chain",
            "chainId",
        ),
        default="",
    )

    if not isinstance(chain_value, str):
        chain_value = get_value(
            chain_value,
            ("chain_id", "id", "name"),
            default=chain_value,
        )

    return normalize_text(chain_value)


def get_atom_serial(atom: AtomLike) -> Optional[Union[int, str]]:
    """
    Return an atom serial number or identifier.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Optional[Union[int, str]]
        Atom serial, identifier, or ``None``.
    """

    value = get_value(
        atom,
        (
            "serial_number",
            "serial",
            "index",
            "id",
            "atom_id",
        ),
        default=None,
    )

    if value is None:
        return None

    integer_value = safe_int(value)

    if integer_value is not None:
        return integer_value

    normalized_value = normalize_text(value)

    return normalized_value or None


def normalize_coordinate(
    coordinate: Any,
    *,
    strict: bool = False,
) -> Optional[Coordinate]:
    """
    Convert a coordinate-like object into a three-component float tuple.

    Parameters
    ----------
    coordinate
        Coordinate-like object, sequence, array, or object exposing x, y, z.
    strict
        Whether invalid coordinates should raise an exception.

    Returns
    -------
    Optional[Coordinate]
        Normalized coordinate or ``None``.

    Raises
    ------
    MissingCoordinatesError
        If strict mode is enabled and the coordinate cannot be normalized.
    """

    if coordinate is None:
        if strict:
            raise MissingCoordinatesError(
                "Coordinate data are unavailable."
            )

        return None

    xyz_values: Any = None

    if all(hasattr(coordinate, axis) for axis in ("x", "y", "z")):
        xyz_values = (
            safe_getattr(coordinate, "x"),
            safe_getattr(coordinate, "y"),
            safe_getattr(coordinate, "z"),
        )
    else:
        try:
            xyz_values = tuple(coordinate)
        except (TypeError, ValueError):
            xyz_values = None

    if xyz_values is None or len(xyz_values) != 3:
        if strict:
            raise MissingCoordinatesError(
                "Coordinates must contain exactly three components."
            )

        return None

    normalized_values = tuple(
        safe_float(value)
        for value in xyz_values
    )

    if any(value is None for value in normalized_values):
        if strict:
            raise MissingCoordinatesError(
                "Coordinates must contain finite numeric values."
            )

        return None

    return (
        float(normalized_values[0]),
        float(normalized_values[1]),
        float(normalized_values[2]),
    )


def get_atom_coordinate(
    atom: AtomLike,
    *,
    strict: bool = False,
) -> Optional[Coordinate]:
    """
    Return normalized Cartesian coordinates for an atom.

    Several common molecular APIs are supported, including ``coord``,
    ``coords``, ``scene_coord``, ``xyz``, and direct x/y/z attributes.

    Parameters
    ----------
    atom
        Molecular atom-like object.
    strict
        Whether missing or invalid coordinates should raise an exception.

    Returns
    -------
    Optional[Coordinate]
        Atom coordinate or ``None``.
    """

    coordinate = get_value(
        atom,
        (
            "scene_coord",
            "coord",
            "coords",
            "coordinate",
            "coordinates",
            "xyz",
        ),
        default=_MISSING,
        call=True,
    )

    if coordinate is _MISSING:
        if all(
            get_value(atom, axis, default=_MISSING) is not _MISSING
            for axis in ("x", "y", "z")
        ):
            coordinate = (
                get_value(atom, "x"),
                get_value(atom, "y"),
                get_value(atom, "z"),
            )
        else:
            coordinate = None

    return normalize_coordinate(
        coordinate,
        strict=strict,
    )


def coordinate_is_finite(coordinate: Any) -> bool:
    """
    Return whether a coordinate can be normalized to finite x, y, z values.

    Parameters
    ----------
    coordinate
        Coordinate-like value.

    Returns
    -------
    bool
        ``True`` when the coordinate is valid.
    """

    return normalize_coordinate(coordinate) is not None


def squared_distance(
    first: Coordinate,
    second: Coordinate,
) -> float:
    """
    Return the squared Euclidean distance between two coordinates.

    Parameters
    ----------
    first
        First Cartesian coordinate.
    second
        Second Cartesian coordinate.

    Returns
    -------
    float
        Squared distance in square angstroms.
    """

    first_coordinate = normalize_coordinate(first, strict=True)
    second_coordinate = normalize_coordinate(second, strict=True)

    return (
        (first_coordinate[0] - second_coordinate[0]) ** 2
        + (first_coordinate[1] - second_coordinate[1]) ** 2
        + (first_coordinate[2] - second_coordinate[2]) ** 2
    )


def distance(
    first: Coordinate,
    second: Coordinate,
) -> float:
    """
    Return the Euclidean distance between two Cartesian coordinates.

    Parameters
    ----------
    first
        First coordinate.
    second
        Second coordinate.

    Returns
    -------
    float
        Distance in angstroms.
    """

    return math.sqrt(squared_distance(first, second))


def mean_coordinate(
    coordinates: Iterable[Coordinate],
    *,
    strict: bool = True,
) -> Optional[Coordinate]:
    """
    Return the arithmetic mean of valid Cartesian coordinates.

    Parameters
    ----------
    coordinates
        Iterable of coordinate-like values.
    strict
        Whether invalid entries or an empty collection should raise an
        exception.

    Returns
    -------
    Optional[Coordinate]
        Mean coordinate or ``None`` when strict mode is disabled.

    Raises
    ------
    DegenerateGeometryError
        If no valid coordinate is available.
    MissingCoordinatesError
        If strict mode is enabled and an invalid coordinate is encountered.
    """

    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    for coordinate in coordinates:
        normalized = normalize_coordinate(
            coordinate,
            strict=strict,
        )

        if normalized is None:
            continue

        sum_x += normalized[0]
        sum_y += normalized[1]
        sum_z += normalized[2]
        count += 1

    if count == 0:
        if strict:
            raise DegenerateGeometryError(
                "A mean coordinate cannot be calculated without "
                "valid coordinates."
            )

        return None

    return (
        sum_x / count,
        sum_y / count,
        sum_z / count,
    )


def iter_atoms(source: Any) -> Iterator[AtomLike]:
    """
    Iterate over atoms from a structure, residue, atom collection, or iterable.

    Parameters
    ----------
    source
        Molecular source object.

    Yields
    ------
    AtomLike
        Atom-like objects.

    Notes
    -----
    Strings, bytes, and mappings are not treated as atom iterables.
    """

    if source is None:
        return

    atom_collection = get_value(
        source,
        ("atoms", "atom_list", "all_atoms"),
        default=_MISSING,
        call=True,
    )

    if atom_collection is not _MISSING and atom_collection is not source:
        try:
            for atom in atom_collection:
                if atom is not None:
                    yield atom

            return
        except TypeError:
            pass

    if isinstance(source, Mapping):
        return

    if isinstance(source, (str, bytes)):
        return

    try:
        iterator = iter(source)
    except TypeError:
        yield source
        return

    for atom in iterator:
        if atom is not None:
            yield atom


def iter_residues(source: Any) -> Iterator[ResidueLike]:
    """
    Iterate over residues from a structure, residue collection, or iterable.

    Parameters
    ----------
    source
        Molecular source object.

    Yields
    ------
    ResidueLike
        Residue-like objects.
    """

    if source is None:
        return

    residue_collection = get_value(
        source,
        ("residues", "residue_list", "all_residues"),
        default=_MISSING,
        call=True,
    )

    if residue_collection is not _MISSING and residue_collection is not source:
        try:
            for residue in residue_collection:
                if residue is not None:
                    yield residue

            return
        except TypeError:
            pass

    if isinstance(source, Mapping):
        return

    if isinstance(source, (str, bytes)):
        return

    try:
        iterator = iter(source)
    except TypeError:
        yield source
        return

    for residue in iterator:
        if residue is not None:
            yield residue


def atom_identity(atom: AtomLike) -> Tuple[Any, ...]:
    """
    Return a hashable identity key for an atom.

    Explicit serial identifiers are preferred. When unavailable, the key uses
    residue and atom descriptors followed by the Python object identity.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Tuple[Any, ...]
        Hashable atom identity key.
    """

    serial = get_atom_serial(atom)

    if serial is not None:
        return ("serial", serial)

    residue = get_atom_residue(atom)

    return (
        "atom",
        get_chain_id(residue),
        get_residue_number(residue),
        get_residue_name(residue),
        get_atom_name(atom),
        id(atom),
    )


def residue_identity(
    residue: Optional[ResidueLike],
) -> Tuple[Any, ...]:
    """
    Return a hashable identity key for a residue.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    Tuple[Any, ...]
        Hashable residue identity key.
    """

    if residue is None:
        return ("residue", None)

    return (
        "residue",
        get_chain_id(residue),
        get_residue_number(residue),
        get_residue_name(residue),
        id(residue),
    )


def charged_atom_identity(
    charged_atom: ChargedAtom,
) -> Tuple[Any, ...]:
    """
    Return a stable identity key for a ChargedAtom instance.

    Parameters
    ----------
    charged_atom
        Charged atom wrapper.

    Returns
    -------
    Tuple[Any, ...]
        Hashable identity key.
    """

    return atom_identity(charged_atom.atom)


def charged_group_identity(
    group: ChargedGroup,
    *,
    include_polarity: bool = True,
) -> Tuple[Any, ...]:
    """
    Return an order-independent identity key for a charged group.

    Parameters
    ----------
    group
        Charged group.
    include_polarity
        Whether group polarity should be included in the key.

    Returns
    -------
    Tuple[Any, ...]
        Hashable charged-group identity key.
    """

    atom_keys = tuple(
        sorted(
            (
                repr(charged_atom_identity(charged_atom)),
                charged_atom_identity(charged_atom),
            )
            for charged_atom in group.atoms
        )
    )

    normalized_atom_keys = tuple(
        atom_key
        for _, atom_key in atom_keys
    )

    if include_polarity:
        return (
            group.polarity,
            group.group_type,
            normalized_atom_keys,
        )

    return (
        group.group_type,
        normalized_atom_keys,
    )


def make_residue_label(
    residue: Optional[ResidueLike],
    *,
    fallback: str = "unknown_residue",
) -> str:
    """
    Build a compact human-readable residue label.

    Parameters
    ----------
    residue
        Residue-like object.
    fallback
        Label returned when residue information is unavailable.

    Returns
    -------
    str
        Residue label such as ``"A:ASP42"``.
    """

    if residue is None:
        return fallback

    chain_id = get_chain_id(residue)
    residue_name = get_residue_name(residue) or "UNK"
    residue_number = get_residue_number(residue)

    number_text = (
        str(residue_number)
        if residue_number is not None
        else "?"
    )

    core_label = f"{residue_name}{number_text}"

    if chain_id:
        return f"{chain_id}:{core_label}"

    return core_label


def make_atom_label(
    atom: AtomLike,
    *,
    fallback: str = "unknown_atom",
) -> str:
    """
    Build a compact human-readable atom label.

    Parameters
    ----------
    atom
        Atom-like object.
    fallback
        Label returned when atom information is unavailable.

    Returns
    -------
    str
        Atom label such as ``"A:LYS15:NZ"``.
    """

    if atom is None:
        return fallback

    atom_name = get_atom_name(atom) or "?"
    residue_label = make_residue_label(
        get_atom_residue(atom),
        fallback="UNK?",
    )

    return f"{residue_label}:{atom_name}"


def make_group_label(
    group: ChargedGroup,
    *,
    include_atoms: bool = False,
) -> str:
    """
    Build a human-readable charged-group label.

    Parameters
    ----------
    group
        Charged group.
    include_atoms
        Whether atom names should be appended to the label.

    Returns
    -------
    str
        Compact group label.
    """

    residue_label = make_residue_label(group.residue)
    base_label = (
        f"{residue_label}:{group.group_type}:{group.polarity}"
    )

    if not include_atoms:
        return base_label

    atom_names = ",".join(
        get_atom_name(charged_atom.atom) or "?"
        for charged_atom in group.atoms
    )

    return f"{base_label}[{atom_names}]"


def resolve_config(
    config: Optional[SaltBridgeConfig] = None,
) -> SaltBridgeConfig:
    """
    Return a validated configuration instance.

    Parameters
    ----------
    config
        Explicit configuration or ``None``.

    Returns
    -------
    SaltBridgeConfig
        Provided configuration or a fresh copy of the default configuration.

    Raises
    ------
    SaltBridgeConfigurationError
        If the supplied object is not a SaltBridgeConfig instance.
    """

    if config is None:
        return DEFAULT_SALT_BRIDGE_CONFIG.copy_with()

    if not isinstance(config, SaltBridgeConfig):
        raise SaltBridgeConfigurationError(
            "config must be a SaltBridgeConfig instance or None."
        )

    return config


def handle_error(
    error: Exception,
    *,
    config: Optional[SaltBridgeConfig] = None,
    warnings: Optional[List[str]] = None,
    context: Optional[str] = None,
) -> None:
    """
    Apply the configured strict or permissive error-handling strategy.

    Parameters
    ----------
    error
        Exception that occurred.
    config
        Salt-bridge configuration.
    warnings
        Optional list receiving non-fatal warning messages.
    context
        Optional operation description added to the warning.

    Raises
    ------
    Exception
        Re-raises the original exception when strict mode is enabled.
    """

    resolved_config = resolve_config(config)

    if resolved_config.strict:
        raise error

    message = str(error).strip() or error.__class__.__name__

    if context:
        message = f"{context}: {message}"

    if warnings is not None and message not in warnings:
        warnings.append(message)


def unique_preserve_order(
    values: Iterable[Any],
    *,
    key: Optional[Any] = None,
) -> List[Any]:
    """
    Return unique values while preserving their original order.

    Parameters
    ----------
    values
        Input iterable.
    key
        Optional callable used to produce hashable identity keys.

    Returns
    -------
    List[Any]
        Ordered list without duplicate entries.
    """

    result: List[Any] = []
    seen: Set[Any] = set()

    for value in values:
        identity = key(value) if key is not None else value

        try:
            already_seen = identity in seen
        except TypeError:
            identity = repr(identity)
            already_seen = identity in seen

        if already_seen:
            continue

        seen.add(identity)
        result.append(value)

    return result


def pairwise_candidates(
    positive_groups: Iterable[ChargedGroup],
    negative_groups: Iterable[ChargedGroup],
) -> Iterator[Tuple[ChargedGroup, ChargedGroup]]:
    """
    Yield cation-anion candidate pairs without materializing a Cartesian list.

    Parameters
    ----------
    positive_groups
        Iterable containing positively charged groups.
    negative_groups
        Iterable containing negatively charged groups.

    Yields
    ------
    Tuple[ChargedGroup, ChargedGroup]
        Ordered cation-anion pair.

    Raises
    ------
    InvalidChargedGroupError
        If a group has an incompatible polarity.
    """

    negative_group_tuple = tuple(negative_groups)

    for negative_group in negative_group_tuple:
        if not negative_group.is_negative:
            raise InvalidChargedGroupError(
                "All negative-group candidates must have negative polarity."
            )

    for positive_group in positive_groups:
        if not positive_group.is_positive:
            raise InvalidChargedGroupError(
                "All positive-group candidates must have positive polarity."
            )

        for negative_group in negative_group_tuple:
            yield positive_group, negative_group



# =============================================================================
# 7. CHARGED-GROUP RECOGNITION
# =============================================================================


# =============================================================================
# 7.1. CHARGE NORMALIZATION AND READING
# =============================================================================


_PROTEIN_RESIDUE_NAMES = frozenset({
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
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
    "LYN",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
})

_NUCLEIC_ACID_RESIDUE_NAMES = frozenset({
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
    "RA",
    "RC",
    "RG",
    "RU",
})

_POSITIVELY_PROTONATED_HISTIDINES = frozenset({
    "HIP",
    "HSP",
})

_NEUTRAL_HISTIDINES = frozenset({
    "HIS",
    "HID",
    "HIE",
    "HSD",
    "HSE",
})

_NEUTRALIZED_ACIDIC_RESIDUES = frozenset({
    "ASH",
    "GLH",
})

_NEUTRALIZED_BASIC_RESIDUES = frozenset({
    "LYN",
})

_PHOSPHATE_ATOM_NAMES = frozenset({
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O1P",
    "O2P",
    "O3P",
})

_CARBOXYLATE_OXYGEN_NAMES = frozenset({
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OXT",
})

_AMINO_TERMINAL_NAMES = frozenset({
    "N",
    "NT",
    "N1",
})

_CARBOXY_TERMINAL_NAMES = frozenset({
    "C",
    "O",
    "OXT",
    "OT1",
    "OT2",
})

_FORMAL_CHARGE_ATTRIBUTE_NAMES = (
    "formal_charge",
    "formalCharge",
    "charge_formal",
    "integer_charge",
)

_PARTIAL_CHARGE_ATTRIBUTE_NAMES = (
    "partial_charge",
    "partialCharge",
    "charge",
    "atomic_charge",
    "gasteiger_charge",
    "gasteigerCharge",
)

_ATOM_NEIGHBOR_ATTRIBUTE_NAMES = (
    "neighbors",
    "bonded_atoms",
    "bondedAtoms",
)

_ATOM_BOND_ATTRIBUTE_NAMES = (
    "bonds",
    "bond_list",
)

_RESIDUE_ATOM_ATTRIBUTE_NAMES = (
    "atoms",
    "atom_list",
)

_STRUCTURE_RESIDUE_ATTRIBUTE_NAMES = (
    "residues",
    "residue_list",
)


def normalize_charge_value(
    value: Any,
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Normalize a formal or partial charge value.

    Parameters
    ----------
    value
        Numeric charge, numeric string, or library-specific charge object.
    default
        Value returned when the charge cannot be interpreted.

    Returns
    -------
    Optional[float]
        Finite normalized charge or the provided default.
    """

    if value is None:
        return default

    if isinstance(value, str):
        normalized_text = value.strip()

        charge_aliases = {
            "+": 1.0,
            "++": 2.0,
            "+++": 3.0,
            "-": -1.0,
            "--": -2.0,
            "---": -3.0,
            "POSITIVE": 1.0,
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0,
        }

        alias_value = charge_aliases.get(normalized_text.upper())

        if alias_value is not None:
            return alias_value

        if normalized_text.endswith("+"):
            magnitude_text = normalized_text[:-1].strip()

            if not magnitude_text:
                return 1.0

            magnitude = safe_float(magnitude_text)

            if magnitude is not None:
                return abs(magnitude)

        if normalized_text.endswith("-"):
            magnitude_text = normalized_text[:-1].strip()

            if not magnitude_text:
                return -1.0

            magnitude = safe_float(magnitude_text)

            if magnitude is not None:
                return -abs(magnitude)

    nested_value = safe_getattr(
        value,
        ("value", "charge", "formal_charge"),
        default=_MISSING,
    )

    if nested_value is not _MISSING and nested_value is not value:
        value = nested_value

    return safe_float(
        value,
        default=default,
        finite_only=True,
    )


def get_atom_formal_charge(
    atom: AtomLike,
) -> Optional[float]:
    """
    Return an explicit formal charge from an atom-like object.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Optional[float]
        Formal charge or ``None`` when unavailable.
    """

    raw_charge = get_value(
        atom,
        _FORMAL_CHARGE_ATTRIBUTE_NAMES,
        default=None,
        call=True,
    )

    return normalize_charge_value(raw_charge)


def get_atom_partial_charge(
    atom: AtomLike,
) -> Optional[float]:
    """
    Return an explicit partial charge from an atom-like object.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Optional[float]
        Partial charge or ``None`` when unavailable.
    """

    raw_charge = get_value(
        atom,
        _PARTIAL_CHARGE_ATTRIBUTE_NAMES,
        default=None,
        call=True,
    )

    return normalize_charge_value(raw_charge)


def classify_numeric_charge(
    charge: Optional[float],
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> str:
    """
    Classify a numeric charge as positive, negative, or neutral.

    Parameters
    ----------
    charge
        Numeric atomic or group charge.
    positive_threshold
        Minimum positive value accepted as positively charged.
    negative_threshold
        Maximum negative value accepted as negatively charged.

    Returns
    -------
    str
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    """

    if charge is None:
        return "neutral"

    normalized_charge = safe_float(charge)

    if normalized_charge is None:
        return "neutral"

    if normalized_charge >= positive_threshold:
        return "positive"

    if normalized_charge <= negative_threshold:
        return "negative"

    return "neutral"


def get_atom_charge_polarity(
    atom: AtomLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Tuple[str, str, Optional[float], Optional[float]]:
    """
    Determine atomic charge polarity from explicit charge information.

    Formal charge is evaluated before partial charge.

    Parameters
    ----------
    atom
        Molecular atom-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    Tuple[str, str, Optional[float], Optional[float]]
        Polarity, recognition source, formal charge, and partial charge.
    """

    resolved_config = resolve_config(config)

    formal_charge = get_atom_formal_charge(atom)
    partial_charge = get_atom_partial_charge(atom)

    if (
        resolved_config.recognize_formal_charges
        and formal_charge is not None
    ):
        formal_polarity = classify_numeric_charge(
            formal_charge,
            positive_threshold=0.5,
            negative_threshold=-0.5,
        )

        if formal_polarity != "neutral":
            return (
                formal_polarity,
                "formal_charge",
                formal_charge,
                partial_charge,
            )

    if (
        resolved_config.recognize_partial_charges
        and partial_charge is not None
    ):
        partial_polarity = classify_numeric_charge(
            partial_charge,
            positive_threshold=(
                resolved_config.partial_charge_positive_threshold
            ),
            negative_threshold=(
                resolved_config.partial_charge_negative_threshold
            ),
        )

        if partial_polarity != "neutral":
            return (
                partial_polarity,
                "partial_charge",
                formal_charge,
                partial_charge,
            )

    return (
        "neutral",
        "unknown",
        formal_charge,
        partial_charge,
    )


def make_charged_atom(
    atom: AtomLike,
    *,
    polarity: Optional[str] = None,
    source: Optional[str] = None,
    config: Optional[SaltBridgeConfig] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ChargedAtom:
    """
    Create a ChargedAtom wrapper from an original atom object.

    Parameters
    ----------
    atom
        Original molecular atom.
    polarity
        Explicit polarity override.
    source
        Explicit recognition-source override.
    config
        Salt-bridge configuration.
    metadata
        Optional compact metadata mapping.

    Returns
    -------
    ChargedAtom
        Validated charged-atom representation.
    """

    resolved_config = resolve_config(config)

    (
        detected_polarity,
        detected_source,
        formal_charge,
        partial_charge,
    ) = get_atom_charge_polarity(
        atom,
        resolved_config,
    )

    final_polarity = normalize_text(
        polarity if polarity is not None else detected_polarity,
        default="neutral",
        lowercase=True,
    )

    final_source = normalize_text(
        source if source is not None else detected_source,
        default="unknown",
        lowercase=True,
    )

    return ChargedAtom(
        atom=atom,
        coordinate=get_atom_coordinate(
            atom,
            strict=resolved_config.strict,
        ),
        element=get_atom_element(atom),
        name=get_atom_name(atom),
        residue=get_atom_residue(atom),
        formal_charge=formal_charge,
        partial_charge=partial_charge,
        polarity=final_polarity,
        source=final_source,
        metadata=dict(metadata or {}),
    )


def get_atom_neighbors(
    atom: AtomLike,
) -> Tuple[AtomLike, ...]:
    """
    Return atoms directly bonded to an atom.

    Parameters
    ----------
    atom
        Molecular atom-like object.

    Returns
    -------
    Tuple[AtomLike, ...]
        Unique neighboring atoms.
    """

    neighbors = get_value(
        atom,
        _ATOM_NEIGHBOR_ATTRIBUTE_NAMES,
        default=_MISSING,
        call=True,
    )

    if neighbors is not _MISSING:
        try:
            return tuple(
                unique_preserve_order(
                    (
                        neighbor
                        for neighbor in neighbors
                        if neighbor is not None
                    ),
                    key=atom_identity,
                )
            )
        except TypeError:
            pass

    bonds = get_value(
        atom,
        _ATOM_BOND_ATTRIBUTE_NAMES,
        default=(),
        call=True,
    )

    collected_neighbors: List[AtomLike] = []

    try:
        bond_iterator = iter(bonds)
    except TypeError:
        bond_iterator = iter(())

    for bond in bond_iterator:
        bond_atoms = get_value(
            bond,
            ("atoms", "endpoints"),
            default=_MISSING,
            call=True,
        )

        if bond_atoms is not _MISSING:
            try:
                for bonded_atom in bond_atoms:
                    if bonded_atom is not atom and bonded_atom is not None:
                        collected_neighbors.append(bonded_atom)

                continue
            except TypeError:
                pass

        first_atom = get_value(
            bond,
            ("atom1", "first_atom", "a1"),
            default=None,
        )

        second_atom = get_value(
            bond,
            ("atom2", "second_atom", "a2"),
            default=None,
        )

        if first_atom is atom and second_atom is not None:
            collected_neighbors.append(second_atom)

        elif second_atom is atom and first_atom is not None:
            collected_neighbors.append(first_atom)

    return tuple(
        unique_preserve_order(
            collected_neighbors,
            key=atom_identity,
        )
    )


def get_residue_atoms(
    residue: Optional[ResidueLike],
) -> Tuple[AtomLike, ...]:
    """
    Return all atoms associated with a residue.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    Tuple[AtomLike, ...]
        Residue atoms.
    """

    if residue is None:
        return ()

    atom_collection = get_value(
        residue,
        _RESIDUE_ATOM_ATTRIBUTE_NAMES,
        default=(),
        call=True,
    )

    try:
        return tuple(
            atom
            for atom in atom_collection
            if atom is not None
        )
    except TypeError:
        return ()


def get_atom_by_name(
    residue: Optional[ResidueLike],
    atom_name: str,
) -> Optional[AtomLike]:
    """
    Return the first residue atom matching an atom name.

    Parameters
    ----------
    residue
        Residue-like object.
    atom_name
        Target atom name.

    Returns
    -------
    Optional[AtomLike]
        Matching atom or ``None``.
    """

    normalized_target = normalize_text(
        atom_name,
        uppercase=True,
    )

    for atom in get_residue_atoms(residue):
        if normalize_text(
            get_atom_name(atom),
            uppercase=True,
        ) == normalized_target:
            return atom

    return None


def get_atoms_by_names(
    residue: Optional[ResidueLike],
    atom_names: Iterable[str],
) -> Tuple[AtomLike, ...]:
    """
    Return residue atoms matching any requested atom name.

    Parameters
    ----------
    residue
        Residue-like object.
    atom_names
        Accepted atom names.

    Returns
    -------
    Tuple[AtomLike, ...]
        Matching atoms in residue order.
    """

    normalized_names = {
        normalize_text(name, uppercase=True)
        for name in atom_names
    }

    return tuple(
        atom
        for atom in get_residue_atoms(residue)
        if normalize_text(
            get_atom_name(atom),
            uppercase=True,
        ) in normalized_names
    )


def classify_residue_category(
    residue: Optional[ResidueLike],
) -> str:
    """
    Classify a residue as protein, nucleic acid, or ligand-like.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    str
        ``"protein"``, ``"nucleic_acid"``, or ``"ligand"``.
    """

    residue_name = get_residue_name(residue)

    if residue_name in _PROTEIN_RESIDUE_NAMES:
        return "protein"

    if residue_name in _NUCLEIC_ACID_RESIDUE_NAMES:
        return "nucleic_acid"

    polymer_type = normalize_text(
        get_value(
            residue,
            (
                "polymer_type",
                "polymerType",
                "structure_category",
            ),
            default="",
        ),
        lowercase=True,
    )

    if "protein" in polymer_type or "amino" in polymer_type:
        return "protein"

    if (
        "nucleic" in polymer_type
        or "dna" in polymer_type
        or "rna" in polymer_type
    ):
        return "nucleic_acid"

    return "ligand"


def residue_category_is_enabled(
    residue: Optional[ResidueLike],
    config: Optional[SaltBridgeConfig] = None,
) -> bool:
    """
    Return whether a residue category is enabled by the configuration.

    Parameters
    ----------
    residue
        Residue-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    bool
        Whether the residue should be analyzed.
    """

    resolved_config = resolve_config(config)
    category = classify_residue_category(residue)

    if category == "protein":
        return resolved_config.include_protein_groups

    if category == "nucleic_acid":
        return resolved_config.include_nucleic_acid_groups

    return resolved_config.include_ligand_groups


# =============================================================================
# 7.2. STANDARD CATIONIC-GROUP RECOGNITION
# =============================================================================


def recognize_arginine_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize the positively charged arginine guanidinium group.

    Parameters
    ----------
    residue
        Arginine residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized guanidinium group or ``None``.
    """

    resolved_config = resolve_config(config)

    if get_residue_name(residue) != "ARG":
        return None

    atoms = get_atoms_by_names(
        residue,
        ("NE", "CZ", "NH1", "NH2"),
    )

    charged_atoms = tuple(
        make_charged_atom(
            atom,
            polarity="positive",
            source="canonical_residue",
            config=resolved_config,
        )
        for atom in atoms
    )

    if not charged_atoms:
        return None

    return ChargedGroup(
        atoms=charged_atoms,
        polarity="positive",
        group_type="guanidinium",
        center=mean_coordinate(
            (
                charged_atom.coordinate
                for charged_atom in charged_atoms
                if charged_atom.coordinate is not None
            ),
            strict=resolved_config.strict,
        ),
        net_charge=1.0,
        residue=residue,
        representative_atom=next(
            (
                charged_atom
                for charged_atom in charged_atoms
                if charged_atom.name.upper() == "CZ"
            ),
            charged_atoms[0],
        ),
        source="canonical_residue",
        confidence=1.0,
        metadata={
            "residue_name": "ARG",
            "charge_model": "delocalized",
        },
    )


def recognize_lysine_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize the positively charged lysine terminal ammonium group.

    Parameters
    ----------
    residue
        Lysine residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized ammonium group or ``None``.
    """

    resolved_config = resolve_config(config)

    if get_residue_name(residue) != "LYS":
        return None

    atom = get_atom_by_name(residue, "NZ")

    if atom is None:
        return None

    charged_atom = make_charged_atom(
        atom,
        polarity="positive",
        source="canonical_residue",
        config=resolved_config,
    )

    return ChargedGroup(
        atoms=(charged_atom,),
        polarity="positive",
        group_type="ammonium",
        center=charged_atom.coordinate,
        net_charge=1.0,
        residue=residue,
        representative_atom=charged_atom,
        source="canonical_residue",
        confidence=1.0,
        metadata={
            "residue_name": "LYS",
            "charge_model": "localized",
        },
    )


def recognize_histidine_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize a positively protonated histidine imidazolium group.

    Neutral histidine variants are not assigned a positive charge unless
    explicit atomic charges independently support that assignment.

    Parameters
    ----------
    residue
        Histidine-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized imidazolium group or ``None``.
    """

    resolved_config = resolve_config(config)
    residue_name = get_residue_name(residue)

    if not resolved_config.allow_histidine_cations:
        return None

    if residue_name not in _POSITIVELY_PROTONATED_HISTIDINES:
        return None

    atoms = get_atoms_by_names(
        residue,
        ("CG", "ND1", "CD2", "CE1", "NE2"),
    )

    charged_atoms = tuple(
        make_charged_atom(
            atom,
            polarity="positive",
            source="canonical_residue",
            config=resolved_config,
        )
        for atom in atoms
    )

    if not charged_atoms:
        return None

    representative_atom = next(
        (
            charged_atom
            for charged_atom in charged_atoms
            if charged_atom.name.upper() in {"ND1", "NE2"}
        ),
        charged_atoms[0],
    )

    return ChargedGroup(
        atoms=charged_atoms,
        polarity="positive",
        group_type="imidazolium",
        center=mean_coordinate(
            (
                charged_atom.coordinate
                for charged_atom in charged_atoms
                if charged_atom.coordinate is not None
            ),
            strict=resolved_config.strict,
        ),
        net_charge=1.0,
        residue=residue,
        representative_atom=representative_atom,
        source="canonical_residue",
        confidence=1.0,
        metadata={
            "residue_name": residue_name,
            "charge_model": "delocalized",
        },
    )


def recognize_canonical_cationic_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize a canonical cationic protein-residue group.

    Parameters
    ----------
    residue
        Residue-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized group or ``None``.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_canonical_residues:
        return None

    residue_name = get_residue_name(residue)

    recognizers = {
        "ARG": recognize_arginine_group,
        "LYS": recognize_lysine_group,
        "HIP": recognize_histidine_group,
        "HSP": recognize_histidine_group,
    }

    recognizer = recognizers.get(residue_name)

    if recognizer is None:
        return None

    return recognizer(residue, resolved_config)


# =============================================================================
# 7.3. STANDARD ANIONIC-GROUP RECOGNITION
# =============================================================================


def recognize_aspartate_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize the negatively charged aspartate carboxylate group.

    Parameters
    ----------
    residue
        Aspartate residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized carboxylate group or ``None``.
    """

    resolved_config = resolve_config(config)

    if get_residue_name(residue) != "ASP":
        return None

    atoms = get_atoms_by_names(
        residue,
        ("CG", "OD1", "OD2"),
    )

    charged_atoms = tuple(
        make_charged_atom(
            atom,
            polarity="negative",
            source="canonical_residue",
            config=resolved_config,
        )
        for atom in atoms
    )

    if not charged_atoms:
        return None

    representative_atom = next(
        (
            charged_atom
            for charged_atom in charged_atoms
            if charged_atom.name.upper() == "CG"
        ),
        charged_atoms[0],
    )

    return ChargedGroup(
        atoms=charged_atoms,
        polarity="negative",
        group_type="carboxylate",
        center=mean_coordinate(
            (
                charged_atom.coordinate
                for charged_atom in charged_atoms
                if charged_atom.coordinate is not None
            ),
            strict=resolved_config.strict,
        ),
        net_charge=-1.0,
        residue=residue,
        representative_atom=representative_atom,
        source="canonical_residue",
        confidence=1.0,
        metadata={
            "residue_name": "ASP",
            "charge_model": "delocalized",
        },
    )


def recognize_glutamate_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize the negatively charged glutamate carboxylate group.

    Parameters
    ----------
    residue
        Glutamate residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized carboxylate group or ``None``.
    """

    resolved_config = resolve_config(config)

    if get_residue_name(residue) != "GLU":
        return None

    atoms = get_atoms_by_names(
        residue,
        ("CD", "OE1", "OE2"),
    )

    charged_atoms = tuple(
        make_charged_atom(
            atom,
            polarity="negative",
            source="canonical_residue",
            config=resolved_config,
        )
        for atom in atoms
    )

    if not charged_atoms:
        return None

    representative_atom = next(
        (
            charged_atom
            for charged_atom in charged_atoms
            if charged_atom.name.upper() == "CD"
        ),
        charged_atoms[0],
    )

    return ChargedGroup(
        atoms=charged_atoms,
        polarity="negative",
        group_type="carboxylate",
        center=mean_coordinate(
            (
                charged_atom.coordinate
                for charged_atom in charged_atoms
                if charged_atom.coordinate is not None
            ),
            strict=resolved_config.strict,
        ),
        net_charge=-1.0,
        residue=residue,
        representative_atom=representative_atom,
        source="canonical_residue",
        confidence=1.0,
        metadata={
            "residue_name": "GLU",
            "charge_model": "delocalized",
        },
    )


def recognize_canonical_anionic_group(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize a canonical anionic protein-residue group.

    Parameters
    ----------
    residue
        Residue-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Recognized group or ``None``.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_canonical_residues:
        return None

    residue_name = get_residue_name(residue)

    recognizers = {
        "ASP": recognize_aspartate_group,
        "GLU": recognize_glutamate_group,
    }

    recognizer = recognizers.get(residue_name)

    if recognizer is None:
        return None

    return recognizer(residue, resolved_config)


# =============================================================================
# 7.4. LIGAND CHARGED-GROUP RECOGNITION
# =============================================================================


def infer_group_type_from_atoms(
    atoms: Sequence[AtomLike],
    polarity: str,
) -> str:
    """
    Infer a compact chemical-group type from atom elements and connectivity.

    Parameters
    ----------
    atoms
        Atoms defining a charged group.
    polarity
        Expected group polarity.

    Returns
    -------
    str
        Inferred group-type label.
    """

    normalized_polarity = normalize_text(
        polarity,
        lowercase=True,
    )

    element_counts: Dict[str, int] = defaultdict(int)

    for atom in atoms:
        element_counts[get_atom_element(atom)] += 1

    nitrogen_count = element_counts.get("N", 0)
    oxygen_count = element_counts.get("O", 0)
    sulfur_count = element_counts.get("S", 0)
    phosphorus_count = element_counts.get("P", 0)
    carbon_count = element_counts.get("C", 0)

    if normalized_polarity == "positive":
        if nitrogen_count >= 3 and carbon_count >= 1:
            return "guanidinium_like"

        if nitrogen_count >= 2:
            return "protonated_nitrogen_cluster"

        if nitrogen_count == 1:
            return "cationic_nitrogen"

        if phosphorus_count >= 1:
            return "phosphonium"

        if sulfur_count >= 1:
            return "sulfonium"

        return "cationic_group"

    if phosphorus_count >= 1 and oxygen_count >= 2:
        return "phosphate"

    if sulfur_count >= 1 and oxygen_count >= 2:
        return "sulfate_or_sulfonate"

    if carbon_count >= 1 and oxygen_count >= 2:
        return "carboxylate_like"

    if oxygen_count >= 1:
        return "anionic_oxygen"

    if sulfur_count >= 1:
        return "anionic_sulfur"

    return "anionic_group"


def build_formal_charge_components(
    atoms: Iterable[AtomLike],
    *,
    polarity: str,
    config: Optional[SaltBridgeConfig] = None,
) -> List[Tuple[AtomLike, ...]]:
    """
    Build connected components of atoms carrying the requested formal polarity.

    Parameters
    ----------
    atoms
        Input atom collection.
    polarity
        Requested polarity.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[Tuple[AtomLike, ...]]
        Connected charged-atom components.
    """

    resolved_config = resolve_config(config)
    normalized_polarity = normalize_text(
        polarity,
        lowercase=True,
    )

    selected_atoms: List[AtomLike] = []

    for atom in atoms:
        formal_charge = get_atom_formal_charge(atom)

        detected_polarity = classify_numeric_charge(
            formal_charge,
            positive_threshold=0.5,
            negative_threshold=-0.5,
        )

        if detected_polarity == normalized_polarity:
            selected_atoms.append(atom)

    selected_identity_map = {
        atom_identity(atom): atom
        for atom in selected_atoms
    }

    remaining_keys = set(selected_identity_map)
    components: List[Tuple[AtomLike, ...]] = []

    while remaining_keys:
        initial_key = remaining_keys.pop()
        stack = [selected_identity_map[initial_key]]
        component: List[AtomLike] = []

        while stack:
            current_atom = stack.pop()
            current_key = atom_identity(current_atom)

            if any(
                atom_identity(existing_atom) == current_key
                for existing_atom in component
            ):
                continue

            component.append(current_atom)

            for neighbor in get_atom_neighbors(current_atom):
                neighbor_key = atom_identity(neighbor)

                if neighbor_key in remaining_keys:
                    remaining_keys.remove(neighbor_key)
                    stack.append(selected_identity_map[neighbor_key])

        components.append(tuple(component))

    return components


def expand_charged_component(
    charged_atoms: Sequence[AtomLike],
    *,
    polarity: str,
) -> Tuple[AtomLike, ...]:
    """
    Expand explicitly charged atoms to include directly bonded heteroatoms.

    This expansion captures delocalized groups when only one atom carries an
    explicit formal charge in the source format.

    Parameters
    ----------
    charged_atoms
        Atoms with explicit charge.
    polarity
        Group polarity.

    Returns
    -------
    Tuple[AtomLike, ...]
        Expanded atom component.
    """

    normalized_polarity = normalize_text(
        polarity,
        lowercase=True,
    )

    expanded_atoms: List[AtomLike] = list(charged_atoms)

    allowed_elements = (
        {"N", "C", "P", "S"}
        if normalized_polarity == "positive"
        else {"O", "S", "P", "C", "N"}
    )

    for atom in charged_atoms:
        for neighbor in get_atom_neighbors(atom):
            if get_atom_element(neighbor) in allowed_elements:
                expanded_atoms.append(neighbor)

                for second_neighbor in get_atom_neighbors(neighbor):
                    if get_atom_element(second_neighbor) in allowed_elements:
                        expanded_atoms.append(second_neighbor)

    return tuple(
        unique_preserve_order(
            expanded_atoms,
            key=atom_identity,
        )
    )


def recognize_ligand_groups_by_formal_charge(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Recognize ligand charged groups using explicit formal charges.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Recognized cationic and anionic ligand groups.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_formal_charges:
        return []

    residue_atoms = get_residue_atoms(residue)
    groups: List[ChargedGroup] = []

    for polarity in ("positive", "negative"):
        components = build_formal_charge_components(
            residue_atoms,
            polarity=polarity,
            config=resolved_config,
        )

        for component in components:
            expanded_component = expand_charged_component(
                component,
                polarity=polarity,
            )

            charged_atoms = tuple(
                make_charged_atom(
                    atom,
                    polarity=polarity,
                    source="formal_charge",
                    config=resolved_config,
                )
                for atom in expanded_component
            )

            formal_charges = tuple(
                charge
                for charge in (
                    get_atom_formal_charge(atom)
                    for atom in component
                )
                if charge is not None
            )

            net_charge = (
                sum(formal_charges)
                if formal_charges
                else (
                    1.0
                    if polarity == "positive"
                    else -1.0
                )
            )

            groups.append(
                ChargedGroup(
                    atoms=charged_atoms,
                    polarity=polarity,
                    group_type=infer_group_type_from_atoms(
                        expanded_component,
                        polarity,
                    ),
                    center=mean_coordinate(
                        (
                            charged_atom.coordinate
                            for charged_atom in charged_atoms
                            if charged_atom.coordinate is not None
                        ),
                        strict=resolved_config.strict,
                    ),
                    net_charge=net_charge,
                    residue=residue,
                    representative_atom=charged_atoms[0],
                    source="formal_charge",
                    confidence=1.0,
                    metadata={
                        "residue_category": "ligand",
                        "explicitly_charged_atom_count": len(component),
                        "expanded_atom_count": len(expanded_component),
                    },
                )
            )

    return groups


def recognize_ligand_groups_by_partial_charge(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Recognize ligand charged atoms using partial-charge thresholds.

    Partial-charge recognition is conservative and initially produces
    single-atom groups. Later consolidation may merge adjacent atoms that
    belong to the same chemically delocalized group.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Partial-charge-derived groups.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_partial_charges:
        return []

    groups: List[ChargedGroup] = []

    for atom in get_residue_atoms(residue):
        partial_charge = get_atom_partial_charge(atom)

        polarity = classify_numeric_charge(
            partial_charge,
            positive_threshold=(
                resolved_config.partial_charge_positive_threshold
            ),
            negative_threshold=(
                resolved_config.partial_charge_negative_threshold
            ),
        )

        if polarity == "neutral":
            continue

        charged_atom = make_charged_atom(
            atom,
            polarity=polarity,
            source="partial_charge",
            config=resolved_config,
        )

        groups.append(
            ChargedGroup(
                atoms=(charged_atom,),
                polarity=polarity,
                group_type=infer_group_type_from_atoms(
                    (atom,),
                    polarity,
                ),
                center=charged_atom.coordinate,
                net_charge=partial_charge,
                residue=residue,
                representative_atom=charged_atom,
                source="partial_charge",
                confidence=0.65,
                metadata={
                    "residue_category": "ligand",
                    "threshold_based": True,
                },
            )
        )

    return groups


def detect_carboxylate_like_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Detect carboxylate-like groups from local carbon-oxygen connectivity.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Inferred carboxylate-like groups.
    """

    resolved_config = resolve_config(config)
    groups: List[ChargedGroup] = []

    for carbon_atom in get_residue_atoms(residue):
        if get_atom_element(carbon_atom) != "C":
            continue

        oxygen_neighbors = tuple(
            neighbor
            for neighbor in get_atom_neighbors(carbon_atom)
            if get_atom_element(neighbor) == "O"
        )

        if len(oxygen_neighbors) < 2:
            continue

        component_atoms = (carbon_atom,) + oxygen_neighbors

        explicit_negative_evidence = any(
            (
                get_atom_formal_charge(oxygen_atom) is not None
                and get_atom_formal_charge(oxygen_atom) < 0.0
            )
            or (
                get_atom_partial_charge(oxygen_atom) is not None
                and get_atom_partial_charge(oxygen_atom)
                <= resolved_config.partial_charge_negative_threshold
            )
            for oxygen_atom in oxygen_neighbors
        )

        confidence = (
            0.90
            if explicit_negative_evidence
            else 0.60
        )

        if (
            confidence < resolved_config.minimum_recognition_confidence
            and not resolved_config.allow_ambiguous_groups
        ):
            continue

        charged_atoms = tuple(
            make_charged_atom(
                atom,
                polarity="negative",
                source="chemical_inference",
                config=resolved_config,
            )
            for atom in component_atoms
        )

        groups.append(
            ChargedGroup(
                atoms=charged_atoms,
                polarity="negative",
                group_type="carboxylate_like",
                center=mean_coordinate(
                    (
                        charged_atom.coordinate
                        for charged_atom in charged_atoms
                        if charged_atom.coordinate is not None
                    ),
                    strict=resolved_config.strict,
                ),
                net_charge=-1.0 if explicit_negative_evidence else None,
                residue=residue,
                representative_atom=charged_atoms[0],
                source="chemical_inference",
                confidence=confidence,
                metadata={
                    "residue_category": "ligand",
                    "oxygen_count": len(oxygen_neighbors),
                    "explicit_charge_evidence": explicit_negative_evidence,
                },
            )
        )

    return groups


def detect_phosphate_or_sulfonate_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Detect phosphate-, sulfate-, and sulfonate-like anionic groups.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Inferred anionic heteroatom groups.
    """

    resolved_config = resolve_config(config)
    groups: List[ChargedGroup] = []

    for central_atom in get_residue_atoms(residue):
        central_element = get_atom_element(central_atom)

        if central_element not in {"P", "S"}:
            continue

        oxygen_neighbors = tuple(
            neighbor
            for neighbor in get_atom_neighbors(central_atom)
            if get_atom_element(neighbor) == "O"
        )

        minimum_oxygen_count = (
            3
            if central_element == "P"
            else 2
        )

        if len(oxygen_neighbors) < minimum_oxygen_count:
            continue

        component_atoms = (central_atom,) + oxygen_neighbors

        explicit_negative_evidence = any(
            (
                get_atom_formal_charge(oxygen_atom) is not None
                and get_atom_formal_charge(oxygen_atom) < 0.0
            )
            or (
                get_atom_partial_charge(oxygen_atom) is not None
                and get_atom_partial_charge(oxygen_atom)
                <= resolved_config.partial_charge_negative_threshold
            )
            for oxygen_atom in oxygen_neighbors
        )

        confidence = (
            0.95
            if explicit_negative_evidence
            else 0.75
        )

        charged_atoms = tuple(
            make_charged_atom(
                atom,
                polarity="negative",
                source="chemical_inference",
                config=resolved_config,
            )
            for atom in component_atoms
        )

        group_type = (
            "phosphate"
            if central_element == "P"
            else "sulfate_or_sulfonate"
        )

        groups.append(
            ChargedGroup(
                atoms=charged_atoms,
                polarity="negative",
                group_type=group_type,
                center=mean_coordinate(
                    (
                        charged_atom.coordinate
                        for charged_atom in charged_atoms
                        if charged_atom.coordinate is not None
                    ),
                    strict=resolved_config.strict,
                ),
                net_charge=-1.0 if explicit_negative_evidence else None,
                residue=residue,
                representative_atom=charged_atoms[0],
                source="chemical_inference",
                confidence=confidence,
                metadata={
                    "residue_category": "ligand",
                    "central_element": central_element,
                    "oxygen_count": len(oxygen_neighbors),
                    "explicit_charge_evidence": explicit_negative_evidence,
                },
            )
        )

    return groups


def detect_cationic_nitrogen_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Detect ligand cationic nitrogen groups.

    Explicit charge evidence is preferred. A nitrogen without explicit charge
    is retained only when ambiguity is allowed and its local connectivity is
    compatible with a protonated or quaternary nitrogen center.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Inferred cationic nitrogen groups.
    """

    resolved_config = resolve_config(config)
    groups: List[ChargedGroup] = []

    for nitrogen_atom in get_residue_atoms(residue):
        if get_atom_element(nitrogen_atom) != "N":
            continue

        formal_charge = get_atom_formal_charge(nitrogen_atom)
        partial_charge = get_atom_partial_charge(nitrogen_atom)
        neighbors = get_atom_neighbors(nitrogen_atom)

        explicit_positive = (
            formal_charge is not None
            and formal_charge > 0.0
        )

        partial_positive = (
            partial_charge is not None
            and partial_charge
            >= resolved_config.partial_charge_positive_threshold
        )

        quaternary_like = len(neighbors) >= 4

        if explicit_positive:
            confidence = 1.0

        elif partial_positive:
            confidence = 0.75

        elif quaternary_like and resolved_config.allow_ambiguous_groups:
            confidence = 0.55

        else:
            continue

        charged_atom = make_charged_atom(
            nitrogen_atom,
            polarity="positive",
            source="chemical_inference",
            config=resolved_config,
        )

        groups.append(
            ChargedGroup(
                atoms=(charged_atom,),
                polarity="positive",
                group_type="cationic_nitrogen",
                center=charged_atom.coordinate,
                net_charge=(
                    formal_charge
                    if formal_charge is not None
                    else partial_charge
                ),
                residue=residue,
                representative_atom=charged_atom,
                source="chemical_inference",
                confidence=confidence,
                metadata={
                    "residue_category": "ligand",
                    "neighbor_count": len(neighbors),
                    "quaternary_like": quaternary_like,
                },
            )
        )

    return groups


def recognize_ligand_charged_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Recognize charged groups in a ligand-like residue.

    Parameters
    ----------
    residue
        Ligand-like residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Recognized ligand charged groups.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.include_ligand_groups:
        return []

    groups: List[ChargedGroup] = []

    groups.extend(
        recognize_ligand_groups_by_formal_charge(
            residue,
            resolved_config,
        )
    )

    if resolved_config.infer_charge_from_chemistry:
        groups.extend(
            detect_carboxylate_like_groups(
                residue,
                resolved_config,
            )
        )

        groups.extend(
            detect_phosphate_or_sulfonate_groups(
                residue,
                resolved_config,
            )
        )

        groups.extend(
            detect_cationic_nitrogen_groups(
                residue,
                resolved_config,
            )
        )

    groups.extend(
        recognize_ligand_groups_by_partial_charge(
            residue,
            resolved_config,
        )
    )

    return groups


# =============================================================================
# 7.5. FORMAL-CHARGE-BASED RECOGNITION
# =============================================================================


def recognize_single_atom_formal_charge_group(
    atom: AtomLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize a single-atom charged group from explicit formal charge.

    This fallback is useful when residue-level connectivity is unavailable.

    Parameters
    ----------
    atom
        Atom-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Charged group or ``None``.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_formal_charges:
        return None

    formal_charge = get_atom_formal_charge(atom)

    polarity = classify_numeric_charge(
        formal_charge,
        positive_threshold=0.5,
        negative_threshold=-0.5,
    )

    if polarity == "neutral":
        return None

    charged_atom = make_charged_atom(
        atom,
        polarity=polarity,
        source="formal_charge",
        config=resolved_config,
    )

    return ChargedGroup(
        atoms=(charged_atom,),
        polarity=polarity,
        group_type=infer_group_type_from_atoms(
            (atom,),
            polarity,
        ),
        center=charged_atom.coordinate,
        net_charge=formal_charge,
        residue=get_atom_residue(atom),
        representative_atom=charged_atom,
        source="formal_charge",
        confidence=1.0,
        metadata={
            "single_atom_fallback": True,
        },
    )


def recognize_single_atom_partial_charge_group(
    atom: AtomLike,
    config: Optional[SaltBridgeConfig] = None,
) -> Optional[ChargedGroup]:
    """
    Recognize a single-atom group from partial charge.

    Parameters
    ----------
    atom
        Atom-like object.
    config
        Salt-bridge configuration.

    Returns
    -------
    Optional[ChargedGroup]
        Charged group or ``None``.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.recognize_partial_charges:
        return None

    partial_charge = get_atom_partial_charge(atom)

    polarity = classify_numeric_charge(
        partial_charge,
        positive_threshold=(
            resolved_config.partial_charge_positive_threshold
        ),
        negative_threshold=(
            resolved_config.partial_charge_negative_threshold
        ),
    )

    if polarity == "neutral":
        return None

    charged_atom = make_charged_atom(
        atom,
        polarity=polarity,
        source="partial_charge",
        config=resolved_config,
    )

    return ChargedGroup(
        atoms=(charged_atom,),
        polarity=polarity,
        group_type=infer_group_type_from_atoms(
            (atom,),
            polarity,
        ),
        center=charged_atom.coordinate,
        net_charge=partial_charge,
        residue=get_atom_residue(atom),
        representative_atom=charged_atom,
        source="partial_charge",
        confidence=0.65,
        metadata={
            "single_atom_fallback": True,
        },
    )


# =============================================================================
# 7.6. CHARGED-GROUP VALIDATION
# =============================================================================


def estimate_group_charge(
    group: ChargedGroup,
) -> Optional[float]:
    """
    Estimate a charged-group net charge from available atomic values.

    Parameters
    ----------
    group
        Charged group.

    Returns
    -------
    Optional[float]
        Estimated charge or ``None``.
    """

    if group.net_charge is not None:
        return group.net_charge

    formal_charges = tuple(
        charged_atom.formal_charge
        for charged_atom in group.atoms
        if charged_atom.formal_charge is not None
    )

    if formal_charges:
        return float(sum(formal_charges))

    partial_charges = tuple(
        charged_atom.partial_charge
        for charged_atom in group.atoms
        if charged_atom.partial_charge is not None
    )

    if partial_charges:
        return float(sum(partial_charges))

    return None


def group_charge_is_consistent(
    group: ChargedGroup,
) -> bool:
    """
    Return whether estimated charge agrees with group polarity.

    Parameters
    ----------
    group
        Charged group.

    Returns
    -------
    bool
        Whether charge and polarity are consistent.
    """

    estimated_charge = estimate_group_charge(group)

    if estimated_charge is None:
        return True

    if group.is_positive:
        return estimated_charge >= 0.0

    return estimated_charge <= 0.0


def group_atoms_share_residue(
    group: ChargedGroup,
) -> bool:
    """
    Return whether all charged atoms belong to the same residue.

    Parameters
    ----------
    group
        Charged group.

    Returns
    -------
    bool
        Whether all parent residues are compatible.
    """

    residue_keys = {
        residue_identity(charged_atom.residue)
        for charged_atom in group.atoms
        if charged_atom.residue is not None
    }

    return len(residue_keys) <= 1


def validate_charged_group(
    group: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
    *,
    require_coordinates: bool = True,
) -> bool:
    """
    Validate a charged group against recognition requirements.

    Parameters
    ----------
    group
        Charged group.
    config
        Salt-bridge configuration.
    require_coordinates
        Whether at least one valid atom coordinate is required.

    Returns
    -------
    bool
        ``True`` when the group is accepted.

    Raises
    ------
    InvalidChargedGroupError
        If the group is invalid in strict mode.
    """

    resolved_config = resolve_config(config)

    validation_errors: List[str] = []

    if not group.atoms:
        validation_errors.append(
            "The charged group does not contain atoms."
        )

    if group.polarity not in {"positive", "negative"}:
        validation_errors.append(
            "The charged group has an unsupported polarity."
        )

    if not group_atoms_share_residue(group):
        validation_errors.append(
            "The charged-group atoms belong to incompatible residues."
        )

    if not group_charge_is_consistent(group):
        validation_errors.append(
            "The estimated charge is inconsistent with group polarity."
        )

    if not 0.0 <= group.confidence <= 1.0:
        validation_errors.append(
            "Recognition confidence is outside the allowed range."
        )

    if (
        group.confidence
        < resolved_config.minimum_recognition_confidence
        and not resolved_config.allow_ambiguous_groups
    ):
        validation_errors.append(
            "Recognition confidence is below the configured minimum."
        )

    if require_coordinates and not group.coordinates:
        validation_errors.append(
            "The charged group does not contain valid coordinates."
        )

    estimated_charge = estimate_group_charge(group)

    if (
        estimated_charge is not None
        and abs(estimated_charge)
        < resolved_config.minimum_group_charge
        and group.source in {"formal_charge", "partial_charge"}
    ):
        validation_errors.append(
            "The estimated group charge is below the configured minimum."
        )

    if validation_errors:
        error = InvalidChargedGroupError(
            " ".join(validation_errors)
        )

        if resolved_config.strict:
            raise error

        return False

    return True


def validate_charged_groups(
    groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    require_coordinates: bool = True,
    warnings: Optional[List[str]] = None,
) -> List[ChargedGroup]:
    """
    Validate and retain acceptable charged groups.

    Parameters
    ----------
    groups
        Candidate charged groups.
    config
        Salt-bridge configuration.
    require_coordinates
        Whether valid coordinates are required.
    warnings
        Optional warning collector.

    Returns
    -------
    List[ChargedGroup]
        Valid charged groups.
    """

    resolved_config = resolve_config(config)
    valid_groups: List[ChargedGroup] = []

    for group in groups:
        try:
            if validate_charged_group(
                group,
                resolved_config,
                require_coordinates=require_coordinates,
            ):
                valid_groups.append(group)

        except SaltBridgeError as error:
            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context="Charged-group validation failed",
            )

    return valid_groups


# =============================================================================
# 7.7. CHARGED-GROUP CONSOLIDATION AND DEDUPLICATION
# =============================================================================


_RECOGNITION_SOURCE_PRIORITY = {
    "canonical_residue": 5,
    "formal_charge": 4,
    "chemical_inference": 3,
    "partial_charge": 2,
    "unknown": 1,
}


def charged_group_source_priority(
    group: ChargedGroup,
) -> int:
    """
    Return the priority assigned to a charged-group recognition source.

    Parameters
    ----------
    group
        Charged group.

    Returns
    -------
    int
        Source-priority value.
    """

    return _RECOGNITION_SOURCE_PRIORITY.get(
        group.source,
        0,
    )


def groups_atomically_overlap(
    first: ChargedGroup,
    second: ChargedGroup,
) -> bool:
    """
    Return whether two charged groups share at least one original atom.

    Parameters
    ----------
    first
        First charged group.
    second
        Second charged group.

    Returns
    -------
    bool
        Whether the groups overlap.
    """

    first_atom_keys = {
        charged_atom_identity(charged_atom)
        for charged_atom in first.atoms
    }

    second_atom_keys = {
        charged_atom_identity(charged_atom)
        for charged_atom in second.atoms
    }

    return bool(first_atom_keys & second_atom_keys)


def groups_are_duplicates(
    first: ChargedGroup,
    second: ChargedGroup,
) -> bool:
    """
    Return whether two groups represent the same chemical charged feature.

    Parameters
    ----------
    first
        First charged group.
    second
        Second charged group.

    Returns
    -------
    bool
        Whether the groups should be considered duplicates.
    """

    if first.polarity != second.polarity:
        return False

    if residue_identity(first.residue) != residue_identity(second.residue):
        return False

    first_atom_keys = {
        charged_atom_identity(charged_atom)
        for charged_atom in first.atoms
    }

    second_atom_keys = {
        charged_atom_identity(charged_atom)
        for charged_atom in second.atoms
    }

    if first_atom_keys == second_atom_keys:
        return True

    if not first_atom_keys or not second_atom_keys:
        return False

    overlap = first_atom_keys & second_atom_keys

    if not overlap:
        return False

    smaller_size = min(
        len(first_atom_keys),
        len(second_atom_keys),
    )

    overlap_fraction = len(overlap) / smaller_size

    return overlap_fraction >= 0.5


def select_preferred_group(
    first: ChargedGroup,
    second: ChargedGroup,
) -> ChargedGroup:
    """
    Select the preferred representation of two duplicate groups.

    Parameters
    ----------
    first
        First charged group.
    second
        Second charged group.

    Returns
    -------
    ChargedGroup
        Preferred group.
    """

    first_priority = charged_group_source_priority(first)
    second_priority = charged_group_source_priority(second)

    if first_priority != second_priority:
        return (
            first
            if first_priority > second_priority
            else second
        )

    if first.confidence != second.confidence:
        return (
            first
            if first.confidence > second.confidence
            else second
        )

    if first.atom_count != second.atom_count:
        return (
            first
            if first.atom_count > second.atom_count
            else second
        )

    first_has_charge = estimate_group_charge(first) is not None
    second_has_charge = estimate_group_charge(second) is not None

    if first_has_charge != second_has_charge:
        return first if first_has_charge else second

    return first


def deduplicate_charged_groups(
    groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Remove duplicate charged-group representations.

    Parameters
    ----------
    groups
        Candidate charged groups.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Deduplicated groups.
    """

    resolved_config = resolve_config(config)
    group_list = list(groups)

    if not resolved_config.deduplicate_groups:
        return group_list

    retained_groups: List[ChargedGroup] = []

    for candidate_group in group_list:
        duplicate_index: Optional[int] = None

        for index, retained_group in enumerate(retained_groups):
            if groups_are_duplicates(
                candidate_group,
                retained_group,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            retained_groups.append(candidate_group)
            continue

        retained_groups[duplicate_index] = select_preferred_group(
            retained_groups[duplicate_index],
            candidate_group,
        )

    return retained_groups


def assign_group_identifiers(
    groups: Iterable[ChargedGroup],
    *,
    prefix: str = "charged_group",
) -> List[ChargedGroup]:
    """
    Assign deterministic identifiers to charged groups lacking an identifier.

    Parameters
    ----------
    groups
        Charged groups.
    prefix
        Identifier prefix.

    Returns
    -------
    List[ChargedGroup]
        Same group objects with assigned identifiers.
    """

    group_list = list(groups)

    for index, group in enumerate(group_list, start=1):
        if group.group_id:
            continue

        residue_label = make_residue_label(
            group.residue,
            fallback="unknown",
        )

        normalized_residue_label = (
            residue_label
            .replace(":", "_")
            .replace(" ", "_")
        )

        group.group_id = (
            f"{prefix}_{normalized_residue_label}_"
            f"{group.polarity}_{index}"
        )

    return group_list


def consolidate_charged_groups(
    groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> List[ChargedGroup]:
    """
    Validate, deduplicate, sort, and identify charged groups.

    Parameters
    ----------
    groups
        Candidate groups.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    List[ChargedGroup]
        Consolidated charged groups.
    """

    resolved_config = resolve_config(config)

    validated_groups = validate_charged_groups(
        groups,
        resolved_config,
        warnings=warnings,
    )

    deduplicated_groups = deduplicate_charged_groups(
        validated_groups,
        resolved_config,
    )

    deduplicated_groups.sort(
        key=lambda group: (
            group.polarity,
            make_residue_label(group.residue),
            group.group_type,
            -group.confidence,
        )
    )

    return assign_group_identifiers(
        deduplicated_groups,
    )


# =============================================================================
# 7.8. PUBLIC RECOGNITION API
# =============================================================================


def recognize_terminal_groups(
    residues: Sequence[ResidueLike],
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Recognize protein N-terminal and C-terminal charged groups.

    Terminal recognition is conservative and relies on residue order plus
    standard terminal atom names. Explicitly capped or modified termini should
    be represented by their actual formal charges whenever available.

    Parameters
    ----------
    residues
        Ordered protein residues.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Recognized terminal groups.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.allow_terminal_groups:
        return []

    protein_residues = [
        residue
        for residue in residues
        if classify_residue_category(residue) == "protein"
    ]

    if not protein_residues:
        return []

    groups: List[ChargedGroup] = []

    residues_by_chain: Dict[str, List[ResidueLike]] = defaultdict(list)

    for residue in protein_residues:
        residues_by_chain[get_chain_id(residue)].append(residue)

    for chain_residues in residues_by_chain.values():
        if not chain_residues:
            continue

        first_residue = chain_residues[0]
        last_residue = chain_residues[-1]

        n_terminal_atom = next(
            (
                atom
                for atom in get_residue_atoms(first_residue)
                if normalize_text(
                    get_atom_name(atom),
                    uppercase=True,
                ) in _AMINO_TERMINAL_NAMES
            ),
            None,
        )

        if n_terminal_atom is not None:
            explicit_charge = get_atom_formal_charge(n_terminal_atom)

            if explicit_charge is None or explicit_charge >= 0.0:
                charged_atom = make_charged_atom(
                    n_terminal_atom,
                    polarity="positive",
                    source="terminal_inference",
                    config=resolved_config,
                )

                groups.append(
                    ChargedGroup(
                        atoms=(charged_atom,),
                        polarity="positive",
                        group_type="n_terminus",
                        center=charged_atom.coordinate,
                        net_charge=(
                            explicit_charge
                            if explicit_charge is not None
                            else 1.0
                        ),
                        residue=first_residue,
                        representative_atom=charged_atom,
                        source="terminal_inference",
                        confidence=0.80,
                        metadata={
                            "terminal_type": "N",
                        },
                    )
                )

        c_terminal_atoms = get_atoms_by_names(
            last_residue,
            ("C", "O", "OXT", "OT1", "OT2"),
        )

        oxygen_atoms = tuple(
            atom
            for atom in c_terminal_atoms
            if get_atom_element(atom) == "O"
        )

        carbon_atoms = tuple(
            atom
            for atom in c_terminal_atoms
            if get_atom_element(atom) == "C"
        )

        if len(oxygen_atoms) >= 2:
            component_atoms = carbon_atoms[:1] + oxygen_atoms

            charged_atoms = tuple(
                make_charged_atom(
                    atom,
                    polarity="negative",
                    source="terminal_inference",
                    config=resolved_config,
                )
                for atom in component_atoms
            )

            groups.append(
                ChargedGroup(
                    atoms=charged_atoms,
                    polarity="negative",
                    group_type="c_terminus",
                    center=mean_coordinate(
                        (
                            charged_atom.coordinate
                            for charged_atom in charged_atoms
                            if charged_atom.coordinate is not None
                        ),
                        strict=resolved_config.strict,
                    ),
                    net_charge=-1.0,
                    residue=last_residue,
                    representative_atom=charged_atoms[0],
                    source="terminal_inference",
                    confidence=0.80,
                    metadata={
                        "terminal_type": "C",
                    },
                )
            )

    return groups


def recognize_nucleic_acid_phosphate_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
) -> List[ChargedGroup]:
    """
    Recognize negatively charged phosphate groups in nucleic acids.

    Parameters
    ----------
    residue
        Nucleic-acid residue.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[ChargedGroup]
        Recognized phosphate groups.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.include_nucleic_acid_groups:
        return []

    phosphorus_atoms = tuple(
        atom
        for atom in get_residue_atoms(residue)
        if (
            get_atom_element(atom) == "P"
            or normalize_text(
                get_atom_name(atom),
                uppercase=True,
            ) == "P"
        )
    )

    groups: List[ChargedGroup] = []

    for phosphorus_atom in phosphorus_atoms:
        oxygen_neighbors = tuple(
            neighbor
            for neighbor in get_atom_neighbors(phosphorus_atom)
            if get_atom_element(neighbor) == "O"
        )

        if not oxygen_neighbors:
            named_atoms = get_atoms_by_names(
                residue,
                _PHOSPHATE_ATOM_NAMES,
            )

            oxygen_neighbors = tuple(
                atom
                for atom in named_atoms
                if get_atom_element(atom) == "O"
            )

        if len(oxygen_neighbors) < 2:
            continue

        component_atoms = (phosphorus_atom,) + oxygen_neighbors

        charged_atoms = tuple(
            make_charged_atom(
                atom,
                polarity="negative",
                source="nucleic_acid_phosphate",
                config=resolved_config,
            )
            for atom in component_atoms
        )

        groups.append(
            ChargedGroup(
                atoms=charged_atoms,
                polarity="negative",
                group_type="phosphate",
                center=mean_coordinate(
                    (
                        charged_atom.coordinate
                        for charged_atom in charged_atoms
                        if charged_atom.coordinate is not None
                    ),
                    strict=resolved_config.strict,
                ),
                net_charge=-1.0,
                residue=residue,
                representative_atom=charged_atoms[0],
                source="nucleic_acid_phosphate",
                confidence=1.0,
                metadata={
                    "residue_category": "nucleic_acid",
                },
            )
        )

    return groups


def recognize_residue_charged_groups(
    residue: ResidueLike,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> List[ChargedGroup]:
    """
    Recognize all charged groups associated with one residue.

    Parameters
    ----------
    residue
        Residue-like object.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    List[ChargedGroup]
        Recognized charged groups.
    """

    resolved_config = resolve_config(config)

    if not residue_category_is_enabled(
        residue,
        resolved_config,
    ):
        return []

    residue_category = classify_residue_category(residue)
    groups: List[ChargedGroup] = []

    try:
        if residue_category == "protein":
            canonical_cation = recognize_canonical_cationic_group(
                residue,
                resolved_config,
            )

            if canonical_cation is not None:
                groups.append(canonical_cation)

            canonical_anion = recognize_canonical_anionic_group(
                residue,
                resolved_config,
            )

            if canonical_anion is not None:
                groups.append(canonical_anion)

        elif residue_category == "nucleic_acid":
            groups.extend(
                recognize_nucleic_acid_phosphate_groups(
                    residue,
                    resolved_config,
                )
            )

        else:
            groups.extend(
                recognize_ligand_charged_groups(
                    residue,
                    resolved_config,
                )
            )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=warnings,
            context=(
                f"Charge recognition failed for "
                f"{make_residue_label(residue)}"
            ),
        )

    return groups


def recognize_charged_groups(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> Tuple[List[ChargedGroup], List[ChargedGroup]]:
    """
    Recognize all cationic and anionic groups in a molecular source.

    The source may be a structure, a residue collection, a residue, or an atom
    collection. Residue-level recognition is preferred because it supports
    canonical and chemically grouped features. Atom-level charge recognition
    is used only as a fallback when residues cannot be obtained.

    Parameters
    ----------
    source
        Molecular structure, residue collection, or atom collection.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    Tuple[List[ChargedGroup], List[ChargedGroup]]
        Cationic groups followed by anionic groups.
    """

    resolved_config = resolve_config(config)
    collected_groups: List[ChargedGroup] = []

    residue_list = list(iter_residues(source))

    source_looks_like_single_residue = (
        source is not None
        and bool(get_residue_name(source))
        and bool(get_residue_atoms(source))
    )

    if source_looks_like_single_residue:
        residue_list = [source]

    if residue_list:
        for residue in residue_list:
            collected_groups.extend(
                recognize_residue_charged_groups(
                    residue,
                    resolved_config,
                    warnings=warnings,
                )
            )

        collected_groups.extend(
            recognize_terminal_groups(
                residue_list,
                resolved_config,
            )
        )

    else:
        for atom in iter_atoms(source):
            group = recognize_single_atom_formal_charge_group(
                atom,
                resolved_config,
            )

            if group is None:
                group = recognize_single_atom_partial_charge_group(
                    atom,
                    resolved_config,
                )

            if group is not None:
                collected_groups.append(group)

    consolidated_groups = consolidate_charged_groups(
        collected_groups,
        resolved_config,
        warnings=warnings,
    )

    cationic_groups = [
        group
        for group in consolidated_groups
        if group.is_positive
    ]

    anionic_groups = [
        group
        for group in consolidated_groups
        if group.is_negative
    ]

    return cationic_groups, anionic_groups


def recognize_cationic_groups(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> List[ChargedGroup]:
    """
    Recognize only positively charged groups.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    List[ChargedGroup]
        Recognized cationic groups.
    """

    cationic_groups, _ = recognize_charged_groups(
        source,
        config,
        warnings=warnings,
    )

    return cationic_groups


def recognize_anionic_groups(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> List[ChargedGroup]:
    """
    Recognize only negatively charged groups.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    List[ChargedGroup]
        Recognized anionic groups.
    """

    _, anionic_groups = recognize_charged_groups(
        source,
        config,
        warnings=warnings,
    )

    return anionic_groups


def split_charged_groups(
    groups: Iterable[ChargedGroup],
) -> Tuple[List[ChargedGroup], List[ChargedGroup]]:
    """
    Split charged groups into positive and negative collections.

    Parameters
    ----------
    groups
        Charged groups.

    Returns
    -------
    Tuple[List[ChargedGroup], List[ChargedGroup]]
        Cationic groups followed by anionic groups.
    """

    cationic_groups: List[ChargedGroup] = []
    anionic_groups: List[ChargedGroup] = []

    for group in groups:
        if group.is_positive:
            cationic_groups.append(group)

        elif group.is_negative:
            anionic_groups.append(group)

        else:
            raise InvalidChargedGroupError(
                "A charged group cannot have neutral polarity."
            )

    return cationic_groups, anionic_groups


# =============================================================================
# 8. GEOMETRY
# =============================================================================


# =============================================================================
# 8.1. GROUP COORDINATE ACCESS AND CENTER CALCULATION
# =============================================================================


def get_charged_atom_coordinate(
    charged_atom: ChargedAtom,
    *,
    strict: bool = False,
) -> Optional[Coordinate]:
    """
    Return the Cartesian coordinate associated with a charged atom.

    The coordinate stored in the ChargedAtom instance is preferred. When it is
    unavailable, the function attempts to recover the coordinate from the
    original atom object.

    Parameters
    ----------
    charged_atom
        Charged-atom representation.
    strict
        Whether missing or invalid coordinates should raise an exception.

    Returns
    -------
    Optional[Coordinate]
        Normalized Cartesian coordinate or ``None``.
    """

    if not isinstance(charged_atom, ChargedAtom):
        raise SaltBridgeGeometryError(
            "charged_atom must be a ChargedAtom instance."
        )

    if charged_atom.coordinate is not None:
        return normalize_coordinate(
            charged_atom.coordinate,
            strict=strict,
        )

    coordinate = get_atom_coordinate(
        charged_atom.atom,
        strict=strict,
    )

    if coordinate is not None:
        charged_atom.coordinate = coordinate

    return coordinate


def iter_group_coordinates(
    group: ChargedGroup,
    *,
    strict: bool = False,
) -> Iterator[Tuple[ChargedAtom, Coordinate]]:
    """
    Yield charged atoms together with their valid coordinates.

    Parameters
    ----------
    group
        Charged group.
    strict
        Whether missing coordinates should raise an exception.

    Yields
    ------
    Tuple[ChargedAtom, Coordinate]
        Charged atom and normalized coordinate.
    """

    if not isinstance(group, ChargedGroup):
        raise SaltBridgeGeometryError(
            "group must be a ChargedGroup instance."
        )

    for charged_atom in group.atoms:
        coordinate = get_charged_atom_coordinate(
            charged_atom,
            strict=strict,
        )

        if coordinate is not None:
            yield charged_atom, coordinate


def calculate_group_center(
    group: ChargedGroup,
    *,
    refresh: bool = False,
    strict: bool = True,
) -> Optional[Coordinate]:
    """
    Calculate the arithmetic center of a charged group.

    The existing stored center is reused unless ``refresh`` is enabled.
    The calculated value is stored back in the group to avoid repeated work.

    Parameters
    ----------
    group
        Charged group.
    refresh
        Whether an existing center should be recalculated.
    strict
        Whether the absence of valid coordinates should raise an exception.

    Returns
    -------
    Optional[Coordinate]
        Group center or ``None``.
    """

    if not isinstance(group, ChargedGroup):
        raise SaltBridgeGeometryError(
            "group must be a ChargedGroup instance."
        )

    if group.center is not None and not refresh:
        return normalize_coordinate(
            group.center,
            strict=strict,
        )

    coordinates = (
        coordinate
        for _, coordinate in iter_group_coordinates(
            group,
            strict=strict,
        )
    )

    calculated_center = mean_coordinate(
        coordinates,
        strict=strict,
    )

    if calculated_center is not None:
        group.center = calculated_center

    return calculated_center


def resolve_group_center(
    group: ChargedGroup,
    *,
    strict: bool = True,
) -> Optional[Coordinate]:
    """
    Return a valid representative center for a charged group.

    Parameters
    ----------
    group
        Charged group.
    strict
        Whether unavailable center coordinates should raise an exception.

    Returns
    -------
    Optional[Coordinate]
        Representative group center.
    """

    if group.center is not None:
        normalized_center = normalize_coordinate(
            group.center,
            strict=False,
        )

        if normalized_center is not None:
            return normalized_center

    return calculate_group_center(
        group,
        refresh=True,
        strict=strict,
    )


def refresh_group_geometry(
    group: ChargedGroup,
    *,
    strict: bool = True,
) -> ChargedGroup:
    """
    Refresh atom coordinates and the representative center of a group.

    Parameters
    ----------
    group
        Charged group.
    strict
        Whether coordinate failures should raise an exception.

    Returns
    -------
    ChargedGroup
        Same group object with refreshed geometric data.
    """

    for charged_atom in group.atoms:
        coordinate = get_atom_coordinate(
            charged_atom.atom,
            strict=strict,
        )

        if coordinate is not None:
            charged_atom.coordinate = coordinate

    group.center = calculate_group_center(
        group,
        refresh=True,
        strict=strict,
    )

    return group


# =============================================================================
# 8.2. CENTER-TO-CENTER GEOMETRY
# =============================================================================


def calculate_group_center_distance(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    *,
    strict: bool = True,
) -> float:
    """
    Calculate the distance between two charged-group centers.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    strict
        Whether unavailable centers should raise an exception.

    Returns
    -------
    float
        Center-to-center distance in angstroms.

    Raises
    ------
    MissingCoordinatesError
        If one of the group centers cannot be resolved.
    """

    first_center = resolve_group_center(
        first_group,
        strict=strict,
    )

    second_center = resolve_group_center(
        second_group,
        strict=strict,
    )

    if first_center is None or second_center is None:
        raise MissingCoordinatesError(
            "Both charged groups require valid center coordinates."
        )

    return distance(first_center, second_center)


def groups_are_center_neighbors(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    cutoff: float,
    *,
    strict: bool = False,
) -> bool:
    """
    Return whether two group centers are within a distance cutoff.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    cutoff
        Maximum accepted center distance.
    strict
        Whether missing coordinates should raise an exception.

    Returns
    -------
    bool
        Whether the centers are within the cutoff.
    """

    normalized_cutoff = safe_float(cutoff)

    if normalized_cutoff is None or normalized_cutoff <= 0.0:
        raise SaltBridgeGeometryError(
            "The center-distance cutoff must be greater than zero."
        )

    try:
        center_distance = calculate_group_center_distance(
            first_group,
            second_group,
            strict=strict,
        )

    except SaltBridgeGeometryError:
        if strict:
            raise

        return False

    return center_distance <= normalized_cutoff


# =============================================================================
# 8.3. ATOM-PAIR DISTANCE ITERATION
# =============================================================================


def iter_intergroup_atom_distances(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    *,
    cutoff: Optional[float] = None,
    minimum_distance: Optional[float] = None,
    strict: bool = False,
) -> Iterator[Tuple[ChargedAtom, ChargedAtom, float]]:
    """
    Yield atom-pair distances between two charged groups.

    The function operates as a generator and does not create a complete
    distance matrix. Optional lower and upper distance filters are evaluated
    using squared distances before the square root is calculated.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    cutoff
        Optional maximum distance to retain.
    minimum_distance
        Optional minimum distance to retain.
    strict
        Whether invalid coordinates should raise an exception.

    Yields
    ------
    Tuple[ChargedAtom, ChargedAtom, float]
        First atom, second atom, and Euclidean distance.
    """

    maximum_distance = (
        safe_float(cutoff)
        if cutoff is not None
        else None
    )

    lower_distance = (
        safe_float(minimum_distance)
        if minimum_distance is not None
        else None
    )

    if maximum_distance is not None and maximum_distance <= 0.0:
        raise SaltBridgeGeometryError(
            "The atom-distance cutoff must be greater than zero."
        )

    if lower_distance is not None and lower_distance < 0.0:
        raise SaltBridgeGeometryError(
            "The minimum atom distance cannot be negative."
        )

    if (
        maximum_distance is not None
        and lower_distance is not None
        and lower_distance > maximum_distance
    ):
        raise SaltBridgeGeometryError(
            "The minimum atom distance cannot exceed the maximum cutoff."
        )

    maximum_squared = (
        maximum_distance ** 2
        if maximum_distance is not None
        else None
    )

    lower_squared = (
        lower_distance ** 2
        if lower_distance is not None
        else None
    )

    second_coordinates = tuple(
        iter_group_coordinates(
            second_group,
            strict=strict,
        )
    )

    if not second_coordinates:
        if strict:
            raise MissingCoordinatesError(
                "The second charged group has no valid atom coordinates."
            )

        return

    first_coordinate_found = False

    for first_atom, first_coordinate in iter_group_coordinates(
        first_group,
        strict=strict,
    ):
        first_coordinate_found = True

        for second_atom, second_coordinate in second_coordinates:
            pair_squared_distance = squared_distance(
                first_coordinate,
                second_coordinate,
            )

            if (
                maximum_squared is not None
                and pair_squared_distance > maximum_squared
            ):
                continue

            if (
                lower_squared is not None
                and pair_squared_distance < lower_squared
            ):
                continue

            yield (
                first_atom,
                second_atom,
                math.sqrt(pair_squared_distance),
            )

    if not first_coordinate_found and strict:
        raise MissingCoordinatesError(
            "The first charged group has no valid atom coordinates."
        )


def iter_cation_anion_atom_distances(
    cation: ChargedGroup,
    anion: ChargedGroup,
    *,
    cutoff: Optional[float] = None,
    minimum_distance: Optional[float] = None,
    strict: bool = False,
) -> Iterator[Tuple[ChargedAtom, ChargedAtom, float]]:
    """
    Yield atom-pair distances in cation-to-anion order.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    cutoff
        Optional maximum distance to retain.
    minimum_distance
        Optional minimum distance to retain.
    strict
        Whether coordinate failures should raise an exception.

    Yields
    ------
    Tuple[ChargedAtom, ChargedAtom, float]
        Positive atom, negative atom, and distance.
    """

    if not cation.is_positive:
        raise InvalidChargedGroupError(
            "The first group must have positive polarity."
        )

    if not anion.is_negative:
        raise InvalidChargedGroupError(
            "The second group must have negative polarity."
        )

    yield from iter_intergroup_atom_distances(
        cation,
        anion,
        cutoff=cutoff,
        minimum_distance=minimum_distance,
        strict=strict,
    )


# =============================================================================
# 8.4. CLOSEST-CONTACT GEOMETRY
# =============================================================================


def find_closest_atom_pair(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    *,
    strict: bool = True,
) -> Tuple[ChargedAtom, ChargedAtom, float]:
    """
    Find the shortest atom-to-atom distance between two charged groups.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    strict
        Whether missing coordinates should raise an exception.

    Returns
    -------
    Tuple[ChargedAtom, ChargedAtom, float]
        Closest atom pair and corresponding distance.

    Raises
    ------
    DegenerateGeometryError
        If no valid atom pair can be evaluated.
    """

    closest_first_atom: Optional[ChargedAtom] = None
    closest_second_atom: Optional[ChargedAtom] = None
    closest_distance = math.inf

    for first_atom, second_atom, atom_distance in (
        iter_intergroup_atom_distances(
            first_group,
            second_group,
            strict=strict,
        )
    ):
        if atom_distance < closest_distance:
            closest_first_atom = first_atom
            closest_second_atom = second_atom
            closest_distance = atom_distance

    if (
        closest_first_atom is None
        or closest_second_atom is None
        or not math.isfinite(closest_distance)
    ):
        raise DegenerateGeometryError(
            "No valid atom pair was available for distance calculation."
        )

    return (
        closest_first_atom,
        closest_second_atom,
        closest_distance,
    )


def find_closest_cation_anion_pair(
    cation: ChargedGroup,
    anion: ChargedGroup,
    *,
    strict: bool = True,
) -> Tuple[ChargedAtom, ChargedAtom, float]:
    """
    Find the closest positive-negative atom pair.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    strict
        Whether coordinate failures should raise an exception.

    Returns
    -------
    Tuple[ChargedAtom, ChargedAtom, float]
        Positive atom, negative atom, and minimum distance.
    """

    if not cation.is_positive:
        raise InvalidChargedGroupError(
            "The cation group must have positive polarity."
        )

    if not anion.is_negative:
        raise InvalidChargedGroupError(
            "The anion group must have negative polarity."
        )

    positive_atom, negative_atom, minimum_distance = (
        find_closest_atom_pair(
            cation,
            anion,
            strict=strict,
        )
    )

    return (
        positive_atom,
        negative_atom,
        minimum_distance,
    )


def calculate_minimum_atom_distance(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    *,
    strict: bool = True,
) -> float:
    """
    Return the shortest atom-to-atom distance between two groups.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    strict
        Whether missing coordinates should raise an exception.

    Returns
    -------
    float
        Minimum atom distance in angstroms.
    """

    _, _, minimum_distance = find_closest_atom_pair(
        first_group,
        second_group,
        strict=strict,
    )

    return minimum_distance


# =============================================================================
# 8.5. CONTACT COLLECTION AND SUMMARY
# =============================================================================


def collect_atomic_contacts(
    cation: ChargedGroup,
    anion: ChargedGroup,
    *,
    cutoff: float,
    minimum_distance: float = 0.0,
    strict: bool = False,
) -> List[Tuple[ChargedAtom, ChargedAtom, float]]:
    """
    Collect cation-anion atom pairs satisfying a distance interval.

    This function materializes only accepted contacts, not the full distance
    matrix.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    cutoff
        Maximum accepted contact distance.
    minimum_distance
        Minimum accepted contact distance.
    strict
        Whether coordinate failures should raise an exception.

    Returns
    -------
    List[Tuple[ChargedAtom, ChargedAtom, float]]
        Accepted atomic contacts sorted by increasing distance.
    """

    contacts = list(
        iter_cation_anion_atom_distances(
            cation,
            anion,
            cutoff=cutoff,
            minimum_distance=minimum_distance,
            strict=strict,
        )
    )

    contacts.sort(key=lambda contact: contact[2])

    return contacts


def count_atomic_contacts(
    cation: ChargedGroup,
    anion: ChargedGroup,
    *,
    cutoff: float,
    minimum_distance: float = 0.0,
    strict: bool = False,
) -> int:
    """
    Count atom pairs satisfying a distance interval.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    cutoff
        Maximum accepted contact distance.
    minimum_distance
        Minimum accepted contact distance.
    strict
        Whether coordinate failures should raise an exception.

    Returns
    -------
    int
        Number of accepted atom pairs.
    """

    return sum(
        1
        for _ in iter_cation_anion_atom_distances(
            cation,
            anion,
            cutoff=cutoff,
            minimum_distance=minimum_distance,
            strict=strict,
        )
    )


def summarize_atomic_contacts(
    contacts: Iterable[
        Tuple[ChargedAtom, ChargedAtom, float]
    ],
) -> Dict[str, Any]:
    """
    Summarize retained cation-anion atomic contacts.

    Parameters
    ----------
    contacts
        Iterable of positive atom, negative atom, and distance tuples.

    Returns
    -------
    Dict[str, Any]
        Contact count, minimum, maximum, mean, and closest atom pair.
    """

    contact_count = 0
    distance_sum = 0.0
    minimum_distance = math.inf
    maximum_distance = -math.inf

    closest_positive_atom: Optional[ChargedAtom] = None
    closest_negative_atom: Optional[ChargedAtom] = None

    for positive_atom, negative_atom, atom_distance in contacts:
        normalized_distance = safe_float(atom_distance)

        if normalized_distance is None or normalized_distance < 0.0:
            raise SaltBridgeGeometryError(
                "Atomic contact distances must be finite and non-negative."
            )

        contact_count += 1
        distance_sum += normalized_distance

        if normalized_distance < minimum_distance:
            minimum_distance = normalized_distance
            closest_positive_atom = positive_atom
            closest_negative_atom = negative_atom

        if normalized_distance > maximum_distance:
            maximum_distance = normalized_distance

    if contact_count == 0:
        return {
            "contact_count": 0,
            "minimum_distance": None,
            "maximum_distance": None,
            "mean_distance": None,
            "closest_positive_atom": None,
            "closest_negative_atom": None,
        }

    return {
        "contact_count": contact_count,
        "minimum_distance": minimum_distance,
        "maximum_distance": maximum_distance,
        "mean_distance": distance_sum / contact_count,
        "closest_positive_atom": closest_positive_atom,
        "closest_negative_atom": closest_negative_atom,
    }


# =============================================================================
# 8.6. GEOMETRIC VALIDATION
# =============================================================================


def validate_group_pair_polarity(
    cation: ChargedGroup,
    anion: ChargedGroup,
) -> None:
    """
    Validate the polarity and identity of a candidate group pair.

    Parameters
    ----------
    cation
        Expected positively charged group.
    anion
        Expected negatively charged group.

    Raises
    ------
    InvalidInteractionError
        If group polarity is invalid or both references describe the same
        charged feature.
    """

    if not isinstance(cation, ChargedGroup):
        raise InvalidInteractionError(
            "The cation must be a ChargedGroup instance."
        )

    if not isinstance(anion, ChargedGroup):
        raise InvalidInteractionError(
            "The anion must be a ChargedGroup instance."
        )

    if not cation.is_positive:
        raise InvalidInteractionError(
            "The cation group must have positive polarity."
        )

    if not anion.is_negative:
        raise InvalidInteractionError(
            "The anion group must have negative polarity."
        )

    cation_atoms = {
        charged_atom_identity(charged_atom)
        for charged_atom in cation.atoms
    }

    anion_atoms = {
        charged_atom_identity(charged_atom)
        for charged_atom in anion.atoms
    }

    if cation_atoms & anion_atoms:
        raise InvalidInteractionError(
            "A salt-bridge pair cannot share the same original atom."
        )


def evaluate_distance_criteria(
    *,
    center_distance: float,
    minimum_atom_distance: float,
    contact_count: int,
    config: Optional[SaltBridgeConfig] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate geometric salt-bridge acceptance criteria.

    Parameters
    ----------
    center_distance
        Distance between charged-group centers.
    minimum_atom_distance
        Shortest atom-to-atom distance.
    contact_count
        Number of retained atomic contacts.
    config
        Salt-bridge configuration.

    Returns
    -------
    Tuple[bool, Optional[str]]
        Validity flag and rejection reason.
    """

    resolved_config = resolve_config(config)

    normalized_center_distance = safe_float(center_distance)
    normalized_minimum_distance = safe_float(minimum_atom_distance)
    normalized_contact_count = safe_int(contact_count)

    if normalized_center_distance is None:
        return False, "The group-center distance is invalid."

    if normalized_minimum_distance is None:
        return False, "The minimum atom distance is invalid."

    if normalized_contact_count is None or normalized_contact_count < 0:
        return False, "The atomic contact count is invalid."

    if (
        normalized_minimum_distance
        < resolved_config.minimum_contact_distance
    ):
        return (
            False,
            "The minimum atom distance is below the allowed overlap limit.",
        )

    if (
        resolved_config.use_minimum_atom_distance
        and normalized_minimum_distance
        > resolved_config.distance_cutoff
    ):
        return (
            False,
            "The minimum atom distance exceeds the salt-bridge cutoff.",
        )

    if (
        resolved_config.use_center_distance
        and normalized_center_distance
        > resolved_config.center_distance_cutoff
    ):
        return (
            False,
            "The group-center distance exceeds the configured cutoff.",
        )

    if normalized_contact_count < resolved_config.minimum_contact_count:
        return (
            False,
            "The number of atomic contacts is below the configured minimum.",
        )

    return True, None


def candidate_pair_passes_center_prefilter(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
) -> bool:
    """
    Apply a low-cost group-center prefilter to a candidate pair.

    A tolerance equal to the atomic contact cutoff is added to the configured
    center cutoff because large delocalized groups may contain close atoms even
    when their arithmetic centers are farther apart.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    config
        Salt-bridge configuration.

    Returns
    -------
    bool
        Whether the pair should proceed to atom-level evaluation.
    """

    resolved_config = resolve_config(config)

    if not resolved_config.use_center_distance:
        return True

    try:
        center_distance = calculate_group_center_distance(
            cation,
            anion,
            strict=resolved_config.strict,
        )

    except SaltBridgeGeometryError:
        if resolved_config.strict:
            raise

        return False

    prefilter_cutoff = (
        resolved_config.center_distance_cutoff
        + resolved_config.atomic_contact_cutoff
    )

    return center_distance <= prefilter_cutoff


# =============================================================================
# 8.7. COMPLETE SALT-BRIDGE GEOMETRY EVALUATION
# =============================================================================


def evaluate_salt_bridge_geometry(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
) -> SaltBridgeGeometry:
    """
    Evaluate the complete geometry of a cation-anion candidate pair.

    The evaluation calculates:

    - group-center distance;
    - shortest atom-to-atom distance;
    - number of contacts within the atomic contact cutoff;
    - optional mean and maximum contact distances;
    - closest positive-negative atom pair;
    - validity and rejection reason.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    config
        Salt-bridge configuration.

    Returns
    -------
    SaltBridgeGeometry
        Complete geometric evaluation.

    Raises
    ------
    SaltBridgeGeometryError
        If required geometry cannot be calculated.
    """

    resolved_config = resolve_config(config)

    validate_group_pair_polarity(
        cation,
        anion,
    )

    center_distance = calculate_group_center_distance(
        cation,
        anion,
        strict=resolved_config.strict,
    )

    (
        closest_positive_atom,
        closest_negative_atom,
        minimum_atom_distance,
    ) = find_closest_cation_anion_pair(
        cation,
        anion,
        strict=resolved_config.strict,
    )

    contacts = collect_atomic_contacts(
        cation,
        anion,
        cutoff=resolved_config.atomic_contact_cutoff,
        minimum_distance=resolved_config.minimum_contact_distance,
        strict=resolved_config.strict,
    )

    contact_count = len(contacts)

    maximum_atom_distance: Optional[float] = None
    mean_atom_distance: Optional[float] = None

    if resolved_config.calculate_all_contact_distances and contacts:
        contact_summary = summarize_atomic_contacts(contacts)

        maximum_atom_distance = contact_summary[
            "maximum_distance"
        ]

        mean_atom_distance = contact_summary[
            "mean_distance"
        ]

    valid, rejection_reason = evaluate_distance_criteria(
        center_distance=center_distance,
        minimum_atom_distance=minimum_atom_distance,
        contact_count=contact_count,
        config=resolved_config,
    )

    return SaltBridgeGeometry(
        center_distance=center_distance,
        minimum_atom_distance=minimum_atom_distance,
        maximum_atom_distance=maximum_atom_distance,
        mean_atom_distance=mean_atom_distance,
        contact_count=contact_count,
        closest_positive_atom=closest_positive_atom,
        closest_negative_atom=closest_negative_atom,
        valid=valid,
        rejection_reason=rejection_reason,
    )


def try_evaluate_salt_bridge_geometry(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> Optional[SaltBridgeGeometry]:
    """
    Evaluate candidate geometry using strict or permissive error handling.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    Optional[SaltBridgeGeometry]
        Evaluated geometry or ``None`` after a permissively handled failure.
    """

    resolved_config = resolve_config(config)

    try:
        return evaluate_salt_bridge_geometry(
            cation,
            anion,
            resolved_config,
        )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=warnings,
            context=(
                "Salt-bridge geometry evaluation failed for "
                f"{make_group_label(cation)} and "
                f"{make_group_label(anion)}"
            ),
        )

        return None


# =============================================================================
# 8.8. BATCH GEOMETRY ITERATION
# =============================================================================


def iter_geometric_candidates(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> Iterator[
    Tuple[ChargedGroup, ChargedGroup, SaltBridgeGeometry]
]:
    """
    Yield geometrically evaluated cation-anion candidate pairs.

    Candidate pairs are generated lazily. The center prefilter is applied
    before the more expensive atom-level evaluation.

    Parameters
    ----------
    cationic_groups
        Positively charged groups.
    anionic_groups
        Negatively charged groups.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Yields
    ------
    Tuple[ChargedGroup, ChargedGroup, SaltBridgeGeometry]
        Cation, anion, and evaluated geometry.
    """

    resolved_config = resolve_config(config)

    for cation, anion in pairwise_candidates(
        cationic_groups,
        anionic_groups,
    ):
        try:
            if not candidate_pair_passes_center_prefilter(
                cation,
                anion,
                resolved_config,
            ):
                continue

            geometry = try_evaluate_salt_bridge_geometry(
                cation,
                anion,
                resolved_config,
                warnings=warnings,
            )

            if geometry is None:
                continue

            if (
                geometry.valid
                or resolved_config.preserve_invalid_candidates
            ):
                yield cation, anion, geometry

        except SaltBridgeError as error:
            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=(
                    "Candidate geometry processing failed for "
                    f"{make_group_label(cation)} and "
                    f"{make_group_label(anion)}"
                ),
            )


def evaluate_group_pair_geometry(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
) -> SaltBridgeGeometry:
    """
    Evaluate two oppositely charged groups regardless of input order.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    config
        Salt-bridge configuration.

    Returns
    -------
    SaltBridgeGeometry
        Evaluated geometry.

    Raises
    ------
    InvalidInteractionError
        If the groups do not have opposite polarities.
    """

    if first_group.is_positive and second_group.is_negative:
        cation = first_group
        anion = second_group

    elif first_group.is_negative and second_group.is_positive:
        cation = second_group
        anion = first_group

    else:
        raise InvalidInteractionError(
            "Salt-bridge geometry requires groups with opposite polarities."
        )

    return evaluate_salt_bridge_geometry(
        cation,
        anion,
        config,
    )

# =============================================================================
# 9. CENTRAL DETECTION
# =============================================================================


# =============================================================================
# 9.1. DETECTION INPUT VALIDATION
# =============================================================================


def validate_detection_groups(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
) -> Tuple[List[ChargedGroup], List[ChargedGroup]]:
    """
    Validate charged groups before central salt-bridge detection.

    Parameters
    ----------
    cationic_groups
        Candidate positively charged groups.
    anionic_groups
        Candidate negatively charged groups.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.

    Returns
    -------
    Tuple[List[ChargedGroup], List[ChargedGroup]]
        Validated cationic groups followed by validated anionic groups.
    """

    resolved_config = resolve_config(config)

    validated_cations: List[ChargedGroup] = []
    validated_anions: List[ChargedGroup] = []

    for group in cationic_groups:
        if not isinstance(group, ChargedGroup):
            error = InvalidChargedGroupError(
                "All cationic candidates must be ChargedGroup instances."
            )

            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context="Cationic-group validation failed",
            )

            continue

        if not group.is_positive:
            error = InvalidChargedGroupError(
                "A cationic candidate has non-positive polarity."
            )

            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=make_group_label(group),
            )

            continue

        try:
            if validate_charged_group(
                group,
                resolved_config,
                require_coordinates=True,
            ):
                validated_cations.append(group)

        except SaltBridgeError as error:
            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=(
                    f"Cationic-group validation failed for "
                    f"{make_group_label(group)}"
                ),
            )

    for group in anionic_groups:
        if not isinstance(group, ChargedGroup):
            error = InvalidChargedGroupError(
                "All anionic candidates must be ChargedGroup instances."
            )

            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context="Anionic-group validation failed",
            )

            continue

        if not group.is_negative:
            error = InvalidChargedGroupError(
                "An anionic candidate has non-negative polarity."
            )

            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=make_group_label(group),
            )

            continue

        try:
            if validate_charged_group(
                group,
                resolved_config,
                require_coordinates=True,
            ):
                validated_anions.append(group)

        except SaltBridgeError as error:
            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=(
                    f"Anionic-group validation failed for "
                    f"{make_group_label(group)}"
                ),
            )

    return validated_cations, validated_anions


def normalize_pose_identifier(
    pose_id: Optional[Union[str, int]],
) -> Optional[Union[str, int]]:
    """
    Normalize a docking-pose identifier.

    Parameters
    ----------
    pose_id
        Pose identifier.

    Returns
    -------
    Optional[Union[str, int]]
        Normalized identifier or ``None``.
    """

    if pose_id is None:
        return None

    if isinstance(pose_id, int):
        return pose_id

    normalized_value = normalize_text(pose_id)

    return normalized_value or None


def normalize_model_identifier(
    model_id: Optional[Union[str, int]],
) -> Optional[Union[str, int]]:
    """
    Normalize a molecular-model identifier.

    Parameters
    ----------
    model_id
        Model identifier.

    Returns
    -------
    Optional[Union[str, int]]
        Normalized identifier or ``None``.
    """

    if model_id is None:
        return None

    if isinstance(model_id, int):
        return model_id

    normalized_value = normalize_text(model_id)

    return normalized_value or None


# =============================================================================
# 9.2. INTERACTION CONSTRUCTION
# =============================================================================


def make_interaction_identifier(
    cation: ChargedGroup,
    anion: ChargedGroup,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    index: Optional[int] = None,
) -> str:
    """
    Build a deterministic human-readable interaction identifier.

    The identifier is intended for reporting and serialization. Definitive
    duplicate detection remains the responsibility of Section 11.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional model identifier.
    index
        Optional interaction sequence number.

    Returns
    -------
    str
        Interaction identifier.
    """

    cation_label = make_group_label(cation)
    anion_label = make_group_label(anion)

    normalized_cation_label = (
        cation_label
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "")
        .replace(",", "_")
        .replace(" ", "_")
    )

    normalized_anion_label = (
        anion_label
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "")
        .replace(",", "_")
        .replace(" ", "_")
    )

    identifier_parts = ["salt_bridge"]

    if model_id is not None:
        identifier_parts.append(
            f"model_{str(model_id).replace(' ', '_')}"
        )

    if pose_id is not None:
        identifier_parts.append(
            f"pose_{str(pose_id).replace(' ', '_')}"
        )

    identifier_parts.extend(
        (
            normalized_cation_label,
            normalized_anion_label,
        )
    )

    if index is not None:
        identifier_parts.append(str(int(index)))

    return "__".join(identifier_parts)


def build_salt_bridge_interaction(
    cation: ChargedGroup,
    anion: ChargedGroup,
    geometry: SaltBridgeGeometry,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    interaction_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SaltBridgeInteraction:
    """
    Build a SaltBridgeInteraction from validated groups and geometry.

    This function does not perform final strength classification or scoring.
    Those values retain their neutral initial state until Section 10 is
    applied.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    geometry
        Evaluated interaction geometry.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    interaction_id
        Optional explicit interaction identifier.
    metadata
        Optional compact interaction metadata.

    Returns
    -------
    SaltBridgeInteraction
        Constructed interaction.
    """

    validate_group_pair_polarity(
        cation,
        anion,
    )

    if not isinstance(geometry, SaltBridgeGeometry):
        raise InvalidInteractionError(
            "geometry must be a SaltBridgeGeometry instance."
        )

    normalized_pose_id = normalize_pose_identifier(pose_id)
    normalized_model_id = normalize_model_identifier(model_id)

    final_interaction_id = (
        normalize_text(interaction_id)
        if interaction_id is not None
        else make_interaction_identifier(
            cation,
            anion,
            pose_id=normalized_pose_id,
            model_id=normalized_model_id,
        )
    )

    interaction_metadata = {
        "detection_stage": "central_detection",
        "geometry_valid": geometry.valid,
        "classification_pending": True,
        "scoring_pending": True,
    }

    if metadata:
        interaction_metadata.update(dict(metadata))

    return SaltBridgeInteraction(
        cation=cation,
        anion=anion,
        geometry=geometry,
        interaction_type=SALT_BRIDGE,
        strength=STRENGTH_WEAK,
        score=0.0,
        pose_id=normalized_pose_id,
        model_id=normalized_model_id,
        interaction_id=final_interaction_id,
        metadata=interaction_metadata,
    )


# =============================================================================
# 9.3. SINGLE-PAIR DETECTION
# =============================================================================


def detect_salt_bridge_pair(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    interaction_id: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> Optional[SaltBridgeInteraction]:
    """
    Detect a salt bridge between one cationic and one anionic group.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    interaction_id
        Optional explicit interaction identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    Optional[SaltBridgeInteraction]
        Detected interaction or ``None``.
    """

    resolved_config = resolve_config(config)

    try:
        validate_group_pair_polarity(
            cation,
            anion,
        )

        if not candidate_pair_passes_center_prefilter(
            cation,
            anion,
            resolved_config,
        ):
            return None

        geometry = evaluate_salt_bridge_geometry(
            cation,
            anion,
            resolved_config,
        )

        if (
            not geometry.valid
            and not resolved_config.preserve_invalid_candidates
        ):
            return None

        return build_salt_bridge_interaction(
            cation,
            anion,
            geometry,
            pose_id=pose_id,
            model_id=model_id,
            interaction_id=interaction_id,
            metadata={
                "preserved_invalid_candidate": not geometry.valid,
            },
        )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=warnings,
            context=(
                "Salt-bridge pair detection failed for "
                f"{make_group_label(cation)} and "
                f"{make_group_label(anion)}"
            ),
        )

        return None


def detect_salt_bridge_between_groups(
    first_group: ChargedGroup,
    second_group: ChargedGroup,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> Optional[SaltBridgeInteraction]:
    """
    Detect a salt bridge between oppositely charged groups in any input order.

    Parameters
    ----------
    first_group
        First charged group.
    second_group
        Second charged group.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    Optional[SaltBridgeInteraction]
        Detected interaction or ``None``.
    """

    if first_group.is_positive and second_group.is_negative:
        cation = first_group
        anion = second_group

    elif first_group.is_negative and second_group.is_positive:
        cation = second_group
        anion = first_group

    else:
        error = InvalidInteractionError(
            "Central salt-bridge detection requires opposite polarities."
        )

        handle_error(
            error,
            config=config,
            warnings=warnings,
            context="Salt-bridge pair detection failed",
        )

        return None

    return detect_salt_bridge_pair(
        cation,
        anion,
        config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
    )


# =============================================================================
# 9.4. MULTIPLE-PAIR DETECTION
# =============================================================================


def iter_detected_salt_bridges(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> Iterator[SaltBridgeInteraction]:
    """
    Yield salt bridges detected from cationic and anionic group collections.

    Detection is performed lazily. Candidate generation and geometry
    evaluation do not require materializing the complete Cartesian product.

    Parameters
    ----------
    cationic_groups
        Positively charged groups.
    anionic_groups
        Negatively charged groups.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Yields
    ------
    SaltBridgeInteraction
        Detected interaction.
    """

    resolved_config = resolve_config(config)

    (
        validated_cations,
        validated_anions,
    ) = validate_detection_groups(
        cationic_groups,
        anionic_groups,
        resolved_config,
        warnings=warnings,
    )

    interaction_index = 0

    for cation, anion, geometry in iter_geometric_candidates(
        validated_cations,
        validated_anions,
        resolved_config,
        warnings=warnings,
    ):
        if (
            not geometry.valid
            and not resolved_config.preserve_invalid_candidates
        ):
            continue

        interaction_index += 1

        interaction_id = make_interaction_identifier(
            cation,
            anion,
            pose_id=pose_id,
            model_id=model_id,
            index=interaction_index,
        )

        try:
            yield build_salt_bridge_interaction(
                cation,
                anion,
                geometry,
                pose_id=pose_id,
                model_id=model_id,
                interaction_id=interaction_id,
                metadata={
                    "candidate_index": interaction_index,
                    "preserved_invalid_candidate": not geometry.valid,
                },
            )

        except SaltBridgeError as error:
            handle_error(
                error,
                config=resolved_config,
                warnings=warnings,
                context=(
                    "Interaction construction failed for "
                    f"{make_group_label(cation)} and "
                    f"{make_group_label(anion)}"
                ),
            )


def detect_salt_bridges_from_groups(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> List[SaltBridgeInteraction]:
    """
    Detect all salt bridges from previously recognized charged groups.

    Parameters
    ----------
    cationic_groups
        Positively charged groups.
    anionic_groups
        Negatively charged groups.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    List[SaltBridgeInteraction]
        Detected salt-bridge interactions.
    """

    return list(
        iter_detected_salt_bridges(
            cationic_groups,
            anionic_groups,
            config,
            pose_id=pose_id,
            model_id=model_id,
            warnings=warnings,
        )
    )


# =============================================================================
# 9.5. SOURCE-LEVEL DETECTION
# =============================================================================


def detect_salt_bridges(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Recognize charged groups and detect salt bridges in a molecular source.

    This is the principal source-level detection function. It performs:

    1. charged-group recognition;
    2. group validation and consolidation;
    3. cation-anion candidate generation;
    4. geometric evaluation;
    5. SaltBridgeInteraction construction;
    6. SaltBridgeResult assembly.

    Final classification, scoring, deduplication, grouping, and statistics are
    intentionally left to later sections.

    Parameters
    ----------
    source
        Molecular structure, model, residue collection, or atom collection.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional external warning collector.

    Returns
    -------
    SaltBridgeResult
        Central detection result.
    """

    resolved_config = resolve_config(config)

    local_warnings: List[str] = []

    if warnings is not None:
        local_warnings.extend(warnings)

    normalized_pose_id = normalize_pose_identifier(pose_id)
    normalized_model_id = normalize_model_identifier(model_id)

    try:
        (
            cationic_groups,
            anionic_groups,
        ) = recognize_charged_groups(
            source,
            resolved_config,
            warnings=local_warnings,
        )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=local_warnings,
            context="Charged-group recognition failed",
        )

        cationic_groups = []
        anionic_groups = []

    interactions = detect_salt_bridges_from_groups(
        cationic_groups,
        anionic_groups,
        resolved_config,
        pose_id=normalized_pose_id,
        model_id=normalized_model_id,
        warnings=local_warnings,
    )

    result = SaltBridgeResult(
        interactions=interactions,
        cationic_groups=cationic_groups,
        anionic_groups=anionic_groups,
        statistics={},
        warnings=unique_preserve_order(local_warnings),
        pose_id=normalized_pose_id,
        model_id=normalized_model_id,
        metadata={
            "analysis_stage": "central_detection",
            "classification_completed": False,
            "scoring_completed": False,
            "deduplication_completed": False,
            "grouping_completed": False,
            "statistics_completed": False,
            "recognized_cation_count": len(cationic_groups),
            "recognized_anion_count": len(anionic_groups),
            "raw_interaction_count": len(interactions),
            "preserve_invalid_candidates": (
                resolved_config.preserve_invalid_candidates
            ),
        },
    )

    if warnings is not None:
        warnings.clear()
        warnings.extend(result.warnings)

    return result


# =============================================================================
# 9.6. PRE-RECOGNIZED GROUP RESULT ASSEMBLY
# =============================================================================


def detect_salt_bridges_in_group_collection(
    groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Detect salt bridges from a mixed collection of charged groups.

    Parameters
    ----------
    groups
        Mixed positive and negative charged groups.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Central detection result.
    """

    resolved_config = resolve_config(config)
    local_warnings: List[str] = []

    if warnings is not None:
        local_warnings.extend(warnings)

    group_list = list(groups)

    try:
        consolidated_groups = consolidate_charged_groups(
            group_list,
            resolved_config,
            warnings=local_warnings,
        )

        (
            cationic_groups,
            anionic_groups,
        ) = split_charged_groups(
            consolidated_groups,
        )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=local_warnings,
            context="Charged-group collection processing failed",
        )

        cationic_groups = []
        anionic_groups = []

    interactions = detect_salt_bridges_from_groups(
        cationic_groups,
        anionic_groups,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=local_warnings,
    )

    result = SaltBridgeResult(
        interactions=interactions,
        cationic_groups=cationic_groups,
        anionic_groups=anionic_groups,
        statistics={},
        warnings=unique_preserve_order(local_warnings),
        pose_id=normalize_pose_identifier(pose_id),
        model_id=normalize_model_identifier(model_id),
        metadata={
            "analysis_stage": "central_detection",
            "input_mode": "pre_recognized_groups",
            "classification_completed": False,
            "scoring_completed": False,
            "deduplication_completed": False,
            "grouping_completed": False,
            "statistics_completed": False,
            "input_group_count": len(group_list),
            "recognized_cation_count": len(cationic_groups),
            "recognized_anion_count": len(anionic_groups),
            "raw_interaction_count": len(interactions),
        },
    )

    if warnings is not None:
        warnings.clear()
        warnings.extend(result.warnings)

    return result


# =============================================================================
# 9.7. CROSS-SOURCE DETECTION
# =============================================================================


def detect_salt_bridges_between_sources(
    positive_source: Any,
    negative_source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Detect salt bridges between two molecular sources.

    Only cationic groups from ``positive_source`` and anionic groups from
    ``negative_source`` are used. This is useful for receptor-ligand,
    protein-protein, or chain-chain analyses with a defined direction.

    Parameters
    ----------
    positive_source
        Source providing candidate cationic groups.
    negative_source
        Source providing candidate anionic groups.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Cross-source detection result.
    """

    resolved_config = resolve_config(config)
    local_warnings: List[str] = []

    if warnings is not None:
        local_warnings.extend(warnings)

    positive_cations = recognize_cationic_groups(
        positive_source,
        resolved_config,
        warnings=local_warnings,
    )

    negative_anions = recognize_anionic_groups(
        negative_source,
        resolved_config,
        warnings=local_warnings,
    )

    interactions = detect_salt_bridges_from_groups(
        positive_cations,
        negative_anions,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=local_warnings,
    )

    result = SaltBridgeResult(
        interactions=interactions,
        cationic_groups=positive_cations,
        anionic_groups=negative_anions,
        statistics={},
        warnings=unique_preserve_order(local_warnings),
        pose_id=normalize_pose_identifier(pose_id),
        model_id=normalize_model_identifier(model_id),
        metadata={
            "analysis_stage": "central_detection",
            "input_mode": "directed_cross_source",
            "classification_completed": False,
            "scoring_completed": False,
            "deduplication_completed": False,
            "grouping_completed": False,
            "statistics_completed": False,
            "recognized_cation_count": len(positive_cations),
            "recognized_anion_count": len(negative_anions),
            "raw_interaction_count": len(interactions),
        },
    )

    if warnings is not None:
        warnings.clear()
        warnings.extend(result.warnings)

    return result


def detect_bidirectional_salt_bridges_between_sources(
    first_source: Any,
    second_source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Detect salt bridges between two sources in both charge directions.

    The function evaluates:

    - cations from the first source against anions from the second source;
    - cations from the second source against anions from the first source.

    Internal interactions within either individual source are not evaluated.

    Parameters
    ----------
    first_source
        First molecular source.
    second_source
        Second molecular source.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Bidirectional cross-source detection result.
    """

    resolved_config = resolve_config(config)
    local_warnings: List[str] = []

    if warnings is not None:
        local_warnings.extend(warnings)

    (
        first_cations,
        first_anions,
    ) = recognize_charged_groups(
        first_source,
        resolved_config,
        warnings=local_warnings,
    )

    (
        second_cations,
        second_anions,
    ) = recognize_charged_groups(
        second_source,
        resolved_config,
        warnings=local_warnings,
    )

    first_to_second = detect_salt_bridges_from_groups(
        first_cations,
        second_anions,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=local_warnings,
    )

    second_to_first = detect_salt_bridges_from_groups(
        second_cations,
        first_anions,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=local_warnings,
    )

    all_cations = first_cations + second_cations
    all_anions = first_anions + second_anions
    all_interactions = first_to_second + second_to_first

    for interaction in first_to_second:
        interaction.metadata["cross_source_direction"] = (
            "first_cation_to_second_anion"
        )

    for interaction in second_to_first:
        interaction.metadata["cross_source_direction"] = (
            "second_cation_to_first_anion"
        )

    result = SaltBridgeResult(
        interactions=all_interactions,
        cationic_groups=all_cations,
        anionic_groups=all_anions,
        statistics={},
        warnings=unique_preserve_order(local_warnings),
        pose_id=normalize_pose_identifier(pose_id),
        model_id=normalize_model_identifier(model_id),
        metadata={
            "analysis_stage": "central_detection",
            "input_mode": "bidirectional_cross_source",
            "classification_completed": False,
            "scoring_completed": False,
            "deduplication_completed": False,
            "grouping_completed": False,
            "statistics_completed": False,
            "first_source_cation_count": len(first_cations),
            "first_source_anion_count": len(first_anions),
            "second_source_cation_count": len(second_cations),
            "second_source_anion_count": len(second_anions),
            "first_to_second_interaction_count": len(first_to_second),
            "second_to_first_interaction_count": len(second_to_first),
            "raw_interaction_count": len(all_interactions),
        },
    )

    if warnings is not None:
        warnings.clear()
        warnings.extend(result.warnings)

    return result


# =============================================================================
# 9.8. RESULT FILTERING AND BASIC ACCESS
# =============================================================================


def get_valid_salt_bridges(
    result: SaltBridgeResult,
) -> List[SaltBridgeInteraction]:
    """
    Return geometrically valid interactions from a detection result.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    List[SaltBridgeInteraction]
        Valid interactions.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    return [
        interaction
        for interaction in result.interactions
        if interaction.geometry.valid
    ]


def get_rejected_salt_bridge_candidates(
    result: SaltBridgeResult,
) -> List[SaltBridgeInteraction]:
    """
    Return preserved geometrically rejected candidates.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    List[SaltBridgeInteraction]
        Invalid preserved candidates.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    return [
        interaction
        for interaction in result.interactions
        if not interaction.geometry.valid
    ]


def filter_salt_bridges_by_distance(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    maximum_distance: float,
    minimum_distance: float = 0.0,
    use_center_distance: bool = False,
) -> List[SaltBridgeInteraction]:
    """
    Filter detected interactions by a distance interval.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    maximum_distance
        Maximum accepted distance.
    minimum_distance
        Minimum accepted distance.
    use_center_distance
        Whether center distance should be used instead of minimum atom
        distance.

    Returns
    -------
    List[SaltBridgeInteraction]
        Filtered interactions.
    """

    normalized_maximum = safe_float(maximum_distance)
    normalized_minimum = safe_float(minimum_distance)

    if normalized_maximum is None or normalized_maximum <= 0.0:
        raise SaltBridgeDetectionError(
            "maximum_distance must be greater than zero."
        )

    if normalized_minimum is None or normalized_minimum < 0.0:
        raise SaltBridgeDetectionError(
            "minimum_distance cannot be negative."
        )

    if normalized_minimum > normalized_maximum:
        raise SaltBridgeDetectionError(
            "minimum_distance cannot exceed maximum_distance."
        )

    filtered_interactions: List[SaltBridgeInteraction] = []

    for interaction in interactions:
        selected_distance = (
            interaction.center_distance
            if use_center_distance
            else interaction.distance
        )

        if (
            normalized_minimum
            <= selected_distance
            <= normalized_maximum
        ):
            filtered_interactions.append(interaction)

    return filtered_interactions


# =============================================================================
# 10. CLASSIFICATION AND SCORING
# =============================================================================


# =============================================================================
# 10.1. STRENGTH CLASSIFICATION
# =============================================================================


def classify_salt_bridge_strength(
    geometry: SaltBridgeGeometry,
    config: Optional[SaltBridgeConfig] = None,
) -> str:
    """
    Classify salt-bridge strength from its geometric measurements.

    Classification is primarily based on the minimum atom-to-atom distance:

    - strong: distance less than or equal to strong_distance_cutoff;
    - moderate: distance less than or equal to moderate_distance_cutoff;
    - weak: distance less than or equal to distance_cutoff;
    - rejected: geometrically invalid or outside the configured cutoff.

    Parameters
    ----------
    geometry
        Evaluated salt-bridge geometry.
    config
        Salt-bridge configuration.

    Returns
    -------
    str
        Strength classification.

    Raises
    ------
    SaltBridgeScoringError
        If geometry is invalid or cannot be classified in strict mode.
    """

    resolved_config = resolve_config(config)

    if not isinstance(geometry, SaltBridgeGeometry):
        raise SaltBridgeScoringError(
            "geometry must be a SaltBridgeGeometry instance."
        )

    minimum_distance = safe_float(
        geometry.minimum_atom_distance
    )

    if minimum_distance is None:
        raise SaltBridgeScoringError(
            "Salt-bridge strength cannot be classified without a valid "
            "minimum atom distance."
        )

    if not geometry.valid:
        return STRENGTH_REJECTED

    if minimum_distance <= resolved_config.strong_distance_cutoff:
        return STRENGTH_STRONG

    if minimum_distance <= resolved_config.moderate_distance_cutoff:
        return STRENGTH_MODERATE

    if minimum_distance <= resolved_config.distance_cutoff:
        return STRENGTH_WEAK

    return STRENGTH_REJECTED


def classify_interaction_strength(
    interaction: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
    *,
    update: bool = True,
) -> str:
    """
    Classify the strength of a salt-bridge interaction.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    config
        Salt-bridge configuration.
    update
        Whether the interaction object should be updated in place.

    Returns
    -------
    str
        Strength classification.
    """

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeScoringError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    strength = classify_salt_bridge_strength(
        interaction.geometry,
        config,
    )

    if update:
        interaction.strength = strength
        interaction.metadata["classification_pending"] = False
        interaction.metadata["classification_completed"] = True
        interaction.metadata["classification_method"] = (
            "minimum_atom_distance"
        )

    return strength


# =============================================================================
# 10.2. BASE SCORE CALCULATION
# =============================================================================


def get_strength_base_score(
    strength: str,
    config: Optional[SaltBridgeConfig] = None,
) -> float:
    """
    Return the configured base score for a strength classification.

    Parameters
    ----------
    strength
        Strength classification.
    config
        Salt-bridge configuration.

    Returns
    -------
    float
        Base score.

    Raises
    ------
    SaltBridgeScoringError
        If the strength label is unsupported.
    """

    resolved_config = resolve_config(config)

    normalized_strength = normalize_text(
        strength,
        lowercase=True,
    )

    score_map = {
        STRENGTH_STRONG: resolved_config.strong_score,
        STRENGTH_MODERATE: resolved_config.moderate_score,
        STRENGTH_WEAK: resolved_config.weak_score,
        STRENGTH_REJECTED: 0.0,
    }

    if normalized_strength not in score_map:
        raise SaltBridgeScoringError(
            f"Unsupported salt-bridge strength: {strength!r}."
        )

    return float(score_map[normalized_strength])


def calculate_distance_quality_factor(
    geometry: SaltBridgeGeometry,
    config: Optional[SaltBridgeConfig] = None,
) -> float:
    """
    Calculate a continuous distance-quality factor.

    The factor varies from 0.0 to 1.0. Shorter distances receive larger values,
    while distances approaching the configured maximum cutoff receive values
    closer to zero.

    The factor is intended as a secondary refinement and does not replace the
    categorical strong, moderate, or weak base score.

    Parameters
    ----------
    geometry
        Salt-bridge geometry.
    config
        Salt-bridge configuration.

    Returns
    -------
    float
        Distance-quality factor between 0.0 and 1.0.
    """

    resolved_config = resolve_config(config)

    minimum_distance = safe_float(
        geometry.minimum_atom_distance
    )

    if minimum_distance is None:
        return 0.0

    lower_bound = resolved_config.minimum_contact_distance
    upper_bound = resolved_config.distance_cutoff

    if upper_bound <= lower_bound:
        return 0.0

    if minimum_distance <= lower_bound:
        return 1.0

    if minimum_distance >= upper_bound:
        return 0.0

    factor = (
        upper_bound - minimum_distance
    ) / (
        upper_bound - lower_bound
    )

    return max(0.0, min(1.0, factor))


# =============================================================================
# 10.3. CONTACT-COUNT BONUS
# =============================================================================


def calculate_contact_count_bonus(
    geometry: SaltBridgeGeometry,
    config: Optional[SaltBridgeConfig] = None,
) -> float:
    """
    Calculate the score bonus derived from multiple atomic contacts.

    Only contacts beyond the configured minimum contact count contribute to
    the bonus.

    Parameters
    ----------
    geometry
        Salt-bridge geometry.
    config
        Salt-bridge configuration.

    Returns
    -------
    float
        Contact-count bonus.
    """

    resolved_config = resolve_config(config)

    contact_count = safe_int(
        geometry.contact_count,
        default=0,
    )

    if contact_count is None:
        contact_count = 0

    additional_contacts = max(
        0,
        contact_count - resolved_config.minimum_contact_count,
    )

    raw_bonus = (
        additional_contacts
        * resolved_config.contact_count_bonus
    )

    return min(
        raw_bonus,
        resolved_config.maximum_contact_bonus,
    )


# =============================================================================
# 10.4. RECOGNITION-CONFIDENCE FACTOR
# =============================================================================


def calculate_group_confidence_factor(
    cation: ChargedGroup,
    anion: ChargedGroup,
) -> float:
    """
    Calculate a joint recognition-confidence factor.

    The geometric mean is used so that one low-confidence group reduces the
    final factor without allowing the other group to fully compensate for it.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.

    Returns
    -------
    float
        Joint confidence factor between 0.0 and 1.0.
    """

    cation_confidence = safe_float(
        cation.confidence,
        default=0.0,
    )

    anion_confidence = safe_float(
        anion.confidence,
        default=0.0,
    )

    cation_confidence = max(
        0.0,
        min(1.0, cation_confidence or 0.0),
    )

    anion_confidence = max(
        0.0,
        min(1.0, anion_confidence or 0.0),
    )

    return math.sqrt(
        cation_confidence
        * anion_confidence
    )


# =============================================================================
# 10.5. CHARGE-MAGNITUDE FACTOR
# =============================================================================


def calculate_group_charge_magnitude(
    group: ChargedGroup,
) -> float:
    """
    Return the absolute estimated charge magnitude of a group.

    Parameters
    ----------
    group
        Charged group.

    Returns
    -------
    float
        Absolute estimated charge magnitude.
    """

    estimated_charge = estimate_group_charge(group)

    if estimated_charge is None:
        return 1.0

    normalized_charge = safe_float(
        estimated_charge,
        default=1.0,
    )

    if normalized_charge is None:
        return 1.0

    return abs(normalized_charge)


def calculate_charge_factor(
    cation: ChargedGroup,
    anion: ChargedGroup,
) -> float:
    """
    Calculate a bounded factor from cation and anion charge magnitudes.

    The geometric mean of both absolute charge magnitudes is calculated and
    limited to the range 0.5 to 2.0.

    Parameters
    ----------
    cation
        Positively charged group.
    anion
        Negatively charged group.

    Returns
    -------
    float
        Charge factor.
    """

    cation_magnitude = calculate_group_charge_magnitude(
        cation
    )

    anion_magnitude = calculate_group_charge_magnitude(
        anion
    )

    factor = math.sqrt(
        cation_magnitude
        * anion_magnitude
    )

    return max(
        0.5,
        min(2.0, factor),
    )


# =============================================================================
# 10.6. COMPLETE INTERACTION SCORE
# =============================================================================


def calculate_salt_bridge_score(
    interaction: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
) -> float:
    """
    Calculate the complete score of a salt-bridge interaction.

    The default score combines:

    1. strength-dependent base score;
    2. atomic contact-count bonus;
    3. optional recognition-confidence weighting;
    4. optional charge-magnitude weighting.

    Rejected or geometrically invalid interactions receive a score of zero.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    config
        Salt-bridge configuration.

    Returns
    -------
    float
        Final non-negative interaction score.
    """

    resolved_config = resolve_config(config)

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeScoringError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    if not resolved_config.scoring_enabled:
        return 0.0

    if not interaction.geometry.valid:
        return 0.0

    strength = classify_salt_bridge_strength(
        interaction.geometry,
        resolved_config,
    )

    if strength == STRENGTH_REJECTED:
        return 0.0

    base_score = get_strength_base_score(
        strength,
        resolved_config,
    )

    contact_bonus = calculate_contact_count_bonus(
        interaction.geometry,
        resolved_config,
    )

    score = base_score + contact_bonus

    confidence_factor = 1.0

    if resolved_config.confidence_weighting:
        confidence_factor = calculate_group_confidence_factor(
            interaction.cation,
            interaction.anion,
        )

        score *= confidence_factor

    charge_factor = 1.0

    if resolved_config.charge_weighting:
        charge_factor = calculate_charge_factor(
            interaction.cation,
            interaction.anion,
        )

        score *= charge_factor

    return max(0.0, float(score))


def build_score_breakdown(
    interaction: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    Build a detailed score-component dictionary.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    config
        Salt-bridge configuration.

    Returns
    -------
    Dict[str, Any]
        Score components and final score.
    """

    resolved_config = resolve_config(config)

    strength = classify_salt_bridge_strength(
        interaction.geometry,
        resolved_config,
    )

    base_score = get_strength_base_score(
        strength,
        resolved_config,
    )

    contact_bonus = calculate_contact_count_bonus(
        interaction.geometry,
        resolved_config,
    )

    distance_quality_factor = calculate_distance_quality_factor(
        interaction.geometry,
        resolved_config,
    )

    confidence_factor = calculate_group_confidence_factor(
        interaction.cation,
        interaction.anion,
    )

    charge_factor = calculate_charge_factor(
        interaction.cation,
        interaction.anion,
    )

    final_score = calculate_salt_bridge_score(
        interaction,
        resolved_config,
    )

    return {
        "strength": strength,
        "base_score": base_score,
        "contact_bonus": contact_bonus,
        "distance_quality_factor": distance_quality_factor,
        "confidence_factor": confidence_factor,
        "confidence_weighting_enabled": (
            resolved_config.confidence_weighting
        ),
        "charge_factor": charge_factor,
        "charge_weighting_enabled": (
            resolved_config.charge_weighting
        ),
        "final_score": final_score,
    }


# =============================================================================
# 10.7. INTERACTION UPDATE
# =============================================================================


def classify_and_score_interaction(
    interaction: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
    *,
    update_metadata: bool = True,
) -> SaltBridgeInteraction:
    """
    Classify and score one salt-bridge interaction in place.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    config
        Salt-bridge configuration.
    update_metadata
        Whether score components should be stored in interaction metadata.

    Returns
    -------
    SaltBridgeInteraction
        Updated interaction.
    """

    resolved_config = resolve_config(config)

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeScoringError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    strength = classify_interaction_strength(
        interaction,
        resolved_config,
        update=True,
    )

    score = calculate_salt_bridge_score(
        interaction,
        resolved_config,
    )

    interaction.strength = strength
    interaction.score = score

    interaction.metadata["classification_pending"] = False
    interaction.metadata["classification_completed"] = True
    interaction.metadata["scoring_pending"] = False
    interaction.metadata["scoring_completed"] = True

    if update_metadata:
        interaction.metadata["score_breakdown"] = (
            build_score_breakdown(
                interaction,
                resolved_config,
            )
        )

    return interaction


def try_classify_and_score_interaction(
    interaction: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
    update_metadata: bool = True,
) -> Optional[SaltBridgeInteraction]:
    """
    Classify and score an interaction using configured error handling.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.
    update_metadata
        Whether score components should be stored.

    Returns
    -------
    Optional[SaltBridgeInteraction]
        Updated interaction or ``None`` after a permissively handled failure.
    """

    resolved_config = resolve_config(config)

    try:
        return classify_and_score_interaction(
            interaction,
            resolved_config,
            update_metadata=update_metadata,
        )

    except SaltBridgeError as error:
        handle_error(
            error,
            config=resolved_config,
            warnings=warnings,
            context=(
                "Salt-bridge classification and scoring failed for "
                f"{interaction.interaction_id or 'unknown interaction'}"
            ),
        )

        return None


# =============================================================================
# 10.8. BATCH CLASSIFICATION AND SCORING
# =============================================================================


def classify_and_score_interactions(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
    *,
    warnings: Optional[List[str]] = None,
    preserve_failed: bool = False,
    update_metadata: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Classify and score multiple salt-bridge interactions.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.
    warnings
        Optional warning collector.
    preserve_failed
        Whether interactions that fail classification should be retained.
    update_metadata
        Whether score components should be stored.

    Returns
    -------
    List[SaltBridgeInteraction]
        Classified and scored interactions.
    """

    resolved_config = resolve_config(config)
    processed_interactions: List[SaltBridgeInteraction] = []

    for interaction in interactions:
        processed_interaction = (
            try_classify_and_score_interaction(
                interaction,
                resolved_config,
                warnings=warnings,
                update_metadata=update_metadata,
            )
        )

        if processed_interaction is not None:
            processed_interactions.append(
                processed_interaction
            )

        elif preserve_failed:
            processed_interactions.append(interaction)

    return processed_interactions


def classify_and_score_result(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    in_place: bool = True,
) -> SaltBridgeResult:
    """
    Classify and score all interactions in a SaltBridgeResult.

    Parameters
    ----------
    result
        Salt-bridge detection result.
    config
        Salt-bridge configuration.
    in_place
        Whether the original result should be modified.

    Returns
    -------
    SaltBridgeResult
        Result containing classified and scored interactions.
    """

    resolved_config = resolve_config(config)

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeScoringError(
            "result must be a SaltBridgeResult instance."
        )

    target_result = result

    if not in_place:
        target_result = SaltBridgeResult(
            interactions=list(result.interactions),
            cationic_groups=list(result.cationic_groups),
            anionic_groups=list(result.anionic_groups),
            statistics=dict(result.statistics),
            warnings=list(result.warnings),
            pose_id=result.pose_id,
            model_id=result.model_id,
            metadata=dict(result.metadata),
        )

    target_result.interactions = classify_and_score_interactions(
        target_result.interactions,
        resolved_config,
        warnings=target_result.warnings,
        preserve_failed=resolved_config.preserve_invalid_candidates,
        update_metadata=not resolved_config.compact_results,
    )

    target_result.metadata["classification_completed"] = True
    target_result.metadata["scoring_completed"] = True
    target_result.metadata["classified_interaction_count"] = len(
        target_result.interactions
    )
    target_result.metadata["total_score"] = sum(
        interaction.score
        for interaction in target_result.interactions
    )

    return target_result


# =============================================================================
# 10.9. COMPLETE DETECTION, CLASSIFICATION, AND SCORING
# =============================================================================


def analyze_salt_bridges(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Recognize, detect, classify, and score salt bridges in one source.

    This function combines Sections 7 through 10. Deduplication, grouping,
    statistics, DockModel integration, and serialization remain separate.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Classified and scored salt-bridge result.
    """

    resolved_config = resolve_config(config)

    result = detect_salt_bridges(
        source,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
    )

    return classify_and_score_result(
        result,
        resolved_config,
        in_place=True,
    )


def analyze_salt_bridges_from_groups(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Detect, classify, and score salt bridges from recognized groups.

    Parameters
    ----------
    cationic_groups
        Positively charged groups.
    anionic_groups
        Negatively charged groups.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Classified and scored result.
    """

    resolved_config = resolve_config(config)

    cation_list = list(cationic_groups)
    anion_list = list(anionic_groups)

    interactions = detect_salt_bridges_from_groups(
        cation_list,
        anion_list,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
    )

    result = SaltBridgeResult(
        interactions=interactions,
        cationic_groups=cation_list,
        anionic_groups=anion_list,
        statistics={},
        warnings=list(warnings or []),
        pose_id=normalize_pose_identifier(pose_id),
        model_id=normalize_model_identifier(model_id),
        metadata={
            "analysis_stage": "central_detection",
            "input_mode": "recognized_groups",
            "classification_completed": False,
            "scoring_completed": False,
            "deduplication_completed": False,
            "grouping_completed": False,
            "statistics_completed": False,
            "recognized_cation_count": len(cation_list),
            "recognized_anion_count": len(anion_list),
            "raw_interaction_count": len(interactions),
        },
    )

    return classify_and_score_result(
        result,
        resolved_config,
        in_place=True,
    )


# =============================================================================
# 10.10. SCORE-BASED FILTERING AND SORTING
# =============================================================================


def filter_salt_bridges_by_strength(
    interactions: Iterable[SaltBridgeInteraction],
    strengths: Union[str, Iterable[str]],
) -> List[SaltBridgeInteraction]:
    """
    Filter interactions by strength classification.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    strengths
        Accepted strength or collection of accepted strengths.

    Returns
    -------
    List[SaltBridgeInteraction]
        Matching interactions.
    """

    if isinstance(strengths, str):
        accepted_strengths = {
            normalize_text(strengths, lowercase=True)
        }

    else:
        accepted_strengths = {
            normalize_text(strength, lowercase=True)
            for strength in strengths
        }

    valid_strengths = {
        STRENGTH_STRONG,
        STRENGTH_MODERATE,
        STRENGTH_WEAK,
        STRENGTH_REJECTED,
    }

    unsupported_strengths = (
        accepted_strengths - valid_strengths
    )

    if unsupported_strengths:
        formatted_strengths = ", ".join(
            sorted(unsupported_strengths)
        )

        raise SaltBridgeScoringError(
            f"Unsupported strength classification or classifications: "
            f"{formatted_strengths}."
        )

    return [
        interaction
        for interaction in interactions
        if interaction.strength in accepted_strengths
    ]


def filter_salt_bridges_by_score(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    minimum_score: float = 0.0,
    maximum_score: Optional[float] = None,
) -> List[SaltBridgeInteraction]:
    """
    Filter salt bridges by an inclusive score interval.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    minimum_score
        Minimum accepted score.
    maximum_score
        Optional maximum accepted score.

    Returns
    -------
    List[SaltBridgeInteraction]
        Matching interactions.
    """

    normalized_minimum = safe_float(minimum_score)

    if normalized_minimum is None or normalized_minimum < 0.0:
        raise SaltBridgeScoringError(
            "minimum_score must be finite and non-negative."
        )

    normalized_maximum = None

    if maximum_score is not None:
        normalized_maximum = safe_float(maximum_score)

        if normalized_maximum is None or normalized_maximum < 0.0:
            raise SaltBridgeScoringError(
                "maximum_score must be finite and non-negative."
            )

        if normalized_maximum < normalized_minimum:
            raise SaltBridgeScoringError(
                "maximum_score cannot be smaller than minimum_score."
            )

    filtered_interactions: List[SaltBridgeInteraction] = []

    for interaction in interactions:
        score = safe_float(
            interaction.score,
            default=0.0,
        )

        if score is None or score < normalized_minimum:
            continue

        if (
            normalized_maximum is not None
            and score > normalized_maximum
        ):
            continue

        filtered_interactions.append(interaction)

    return filtered_interactions


def sort_salt_bridges_by_score(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    descending: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Sort salt bridges by score and geometric distance.

    Score is the primary key. Minimum atom distance is used as a secondary key
    so that shorter interactions are preferred when scores are equal.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    descending
        Whether higher scores should appear first.

    Returns
    -------
    List[SaltBridgeInteraction]
        Sorted interactions.
    """

    interaction_list = list(interactions)

    if descending:
        return sorted(
            interaction_list,
            key=lambda interaction: (
                -safe_float(
                    interaction.score,
                    default=0.0,
                ),
                safe_float(
                    interaction.distance,
                    default=math.inf,
                ),
                interaction.interaction_id or "",
            ),
        )

    return sorted(
        interaction_list,
        key=lambda interaction: (
            safe_float(
                interaction.score,
                default=0.0,
            ),
            safe_float(
                interaction.distance,
                default=math.inf,
            ),
            interaction.interaction_id or "",
        ),
    )


def get_best_salt_bridge(
    interactions: Iterable[SaltBridgeInteraction],
) -> Optional[SaltBridgeInteraction]:
    """
    Return the highest-scoring salt bridge.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.

    Returns
    -------
    Optional[SaltBridgeInteraction]
        Best interaction or ``None``.
    """

    sorted_interactions = sort_salt_bridges_by_score(
        interactions,
        descending=True,
    )

    if not sorted_interactions:
        return None

    return sorted_interactions[0]



# =============================================================================
# 11. DEDUPLICATION
# =============================================================================


# =============================================================================
# 11.1. INTERACTION IDENTITY KEYS
# =============================================================================


def interaction_group_pair_key(
    interaction: SaltBridgeInteraction,
    *,
    include_pose: bool = True,
    include_model: bool = True,
    include_group_type: bool = True,
) -> Tuple[Any, ...]:
    """
    Build an identity key from the cation-anion group pair.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    include_pose
        Whether the pose identifier should be included.
    include_model
        Whether the model identifier should be included.
    include_group_type
        Whether charged-group types should be included.

    Returns
    -------
    Tuple[Any, ...]
        Hashable interaction identity key.
    """

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeDetectionError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    cation_key = charged_group_identity(
        interaction.cation,
        include_polarity=True,
    )

    anion_key = charged_group_identity(
        interaction.anion,
        include_polarity=True,
    )

    if not include_group_type:
        cation_key = (
            interaction.cation.polarity,
            tuple(
                sorted(
                    (
                        repr(charged_atom_identity(charged_atom)),
                        charged_atom_identity(charged_atom),
                    )
                    for charged_atom in interaction.cation.atoms
                )
            ),
        )

        anion_key = (
            interaction.anion.polarity,
            tuple(
                sorted(
                    (
                        repr(charged_atom_identity(charged_atom)),
                        charged_atom_identity(charged_atom),
                    )
                    for charged_atom in interaction.anion.atoms
                )
            ),
        )

    key_parts: List[Any] = [
        "salt_bridge",
        cation_key,
        anion_key,
    ]

    if include_pose:
        key_parts.append(
            ("pose", interaction.pose_id)
        )

    if include_model:
        key_parts.append(
            ("model", interaction.model_id)
        )

    return tuple(key_parts)


def interaction_atom_pair_key(
    interaction: SaltBridgeInteraction,
    *,
    include_pose: bool = True,
    include_model: bool = True,
) -> Tuple[Any, ...]:
    """
    Build an identity key from the closest positive-negative atom pair.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    include_pose
        Whether the pose identifier should be included.
    include_model
        Whether the model identifier should be included.

    Returns
    -------
    Tuple[Any, ...]
        Hashable closest-contact identity key.
    """

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeDetectionError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    closest_positive_atom = (
        interaction.geometry.closest_positive_atom
    )

    closest_negative_atom = (
        interaction.geometry.closest_negative_atom
    )

    positive_atom_key = (
        charged_atom_identity(closest_positive_atom)
        if closest_positive_atom is not None
        else None
    )

    negative_atom_key = (
        charged_atom_identity(closest_negative_atom)
        if closest_negative_atom is not None
        else None
    )

    key_parts: List[Any] = [
        "salt_bridge_atom_pair",
        positive_atom_key,
        negative_atom_key,
    ]

    if include_pose:
        key_parts.append(
            ("pose", interaction.pose_id)
        )

    if include_model:
        key_parts.append(
            ("model", interaction.model_id)
        )

    return tuple(key_parts)


def interaction_residue_pair_key(
    interaction: SaltBridgeInteraction,
    *,
    include_pose: bool = True,
    include_model: bool = True,
    include_group_type: bool = False,
) -> Tuple[Any, ...]:
    """
    Build an interaction key from the participating residues.

    Residue-level keys are intentionally broader than group-level keys. They
    are useful when one residue pair generates multiple equivalent atomic
    representations.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    include_pose
        Whether the pose identifier should be included.
    include_model
        Whether the model identifier should be included.
    include_group_type
        Whether cation and anion group types should be included.

    Returns
    -------
    Tuple[Any, ...]
        Hashable residue-pair identity key.
    """

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeDetectionError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    key_parts: List[Any] = [
        "salt_bridge_residue_pair",
        residue_identity(interaction.cation.residue),
        residue_identity(interaction.anion.residue),
    ]

    if include_group_type:
        key_parts.extend(
            (
                interaction.cation.group_type,
                interaction.anion.group_type,
            )
        )

    if include_pose:
        key_parts.append(
            ("pose", interaction.pose_id)
        )

    if include_model:
        key_parts.append(
            ("model", interaction.model_id)
        )

    return tuple(key_parts)


def interaction_identity_key(
    interaction: SaltBridgeInteraction,
    *,
    mode: str = "group_pair",
    include_pose: bool = True,
    include_model: bool = True,
) -> Tuple[Any, ...]:
    """
    Build an interaction identity key using a selected strategy.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    mode
        Identity mode. Supported values are ``"group_pair"``,
        ``"atom_pair"``, and ``"residue_pair"``.
    include_pose
        Whether pose information should be included.
    include_model
        Whether model information should be included.

    Returns
    -------
    Tuple[Any, ...]
        Hashable identity key.

    Raises
    ------
    SaltBridgeDetectionError
        If the identity mode is unsupported.
    """

    normalized_mode = normalize_text(
        mode,
        lowercase=True,
    )

    if normalized_mode == "group_pair":
        return interaction_group_pair_key(
            interaction,
            include_pose=include_pose,
            include_model=include_model,
        )

    if normalized_mode == "atom_pair":
        return interaction_atom_pair_key(
            interaction,
            include_pose=include_pose,
            include_model=include_model,
        )

    if normalized_mode == "residue_pair":
        return interaction_residue_pair_key(
            interaction,
            include_pose=include_pose,
            include_model=include_model,
        )

    raise SaltBridgeDetectionError(
        f"Unsupported interaction identity mode: {mode!r}."
    )


# =============================================================================
# 11.2. ATOMIC AND GROUP OVERLAP
# =============================================================================


def interaction_atom_sets(
    interaction: SaltBridgeInteraction,
) -> Tuple[Set[Tuple[Any, ...]], Set[Tuple[Any, ...]]]:
    """
    Return cationic and anionic atom-identity sets.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Tuple[Set[Tuple[Any, ...]], Set[Tuple[Any, ...]]]
        Cationic atom keys followed by anionic atom keys.
    """

    cation_atoms = {
        charged_atom_identity(charged_atom)
        for charged_atom in interaction.cation.atoms
    }

    anion_atoms = {
        charged_atom_identity(charged_atom)
        for charged_atom in interaction.anion.atoms
    }

    return cation_atoms, anion_atoms


def calculate_set_overlap_fraction(
    first_set: Set[Any],
    second_set: Set[Any],
) -> float:
    """
    Calculate overlap relative to the smaller non-empty set.

    Parameters
    ----------
    first_set
        First identity set.
    second_set
        Second identity set.

    Returns
    -------
    float
        Overlap fraction between 0.0 and 1.0.
    """

    if not first_set or not second_set:
        return 0.0

    overlap_size = len(first_set & second_set)
    smaller_size = min(
        len(first_set),
        len(second_set),
    )

    if smaller_size == 0:
        return 0.0

    return overlap_size / smaller_size


def calculate_interaction_atomic_overlap(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
) -> Tuple[float, float]:
    """
    Calculate cationic and anionic atomic overlap fractions.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.

    Returns
    -------
    Tuple[float, float]
        Cation overlap followed by anion overlap.
    """

    (
        first_cation_atoms,
        first_anion_atoms,
    ) = interaction_atom_sets(first)

    (
        second_cation_atoms,
        second_anion_atoms,
    ) = interaction_atom_sets(second)

    cation_overlap = calculate_set_overlap_fraction(
        first_cation_atoms,
        second_cation_atoms,
    )

    anion_overlap = calculate_set_overlap_fraction(
        first_anion_atoms,
        second_anion_atoms,
    )

    return cation_overlap, anion_overlap


def interactions_share_group_pair(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
) -> bool:
    """
    Return whether two interactions contain the same charged-group pair.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.

    Returns
    -------
    bool
        Whether both group identities match.
    """

    return (
        charged_group_identity(first.cation)
        == charged_group_identity(second.cation)
        and charged_group_identity(first.anion)
        == charged_group_identity(second.anion)
    )


def interactions_share_residue_pair(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
) -> bool:
    """
    Return whether two interactions connect the same residue pair.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.

    Returns
    -------
    bool
        Whether both residue identities match.
    """

    return (
        residue_identity(first.cation.residue)
        == residue_identity(second.cation.residue)
        and residue_identity(first.anion.residue)
        == residue_identity(second.anion.residue)
    )


def interactions_share_context(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
    *,
    include_pose: bool = True,
    include_model: bool = True,
) -> bool:
    """
    Return whether interactions belong to the same pose and model context.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    bool
        Whether the selected contextual identifiers match.
    """

    if include_pose and first.pose_id != second.pose_id:
        return False

    if include_model and first.model_id != second.model_id:
        return False

    return True


# =============================================================================
# 11.3. DUPLICATE DECISION
# =============================================================================


def interactions_are_exact_duplicates(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
    *,
    include_pose: bool = True,
    include_model: bool = True,
) -> bool:
    """
    Return whether two interactions have identical group-pair identities.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    bool
        Whether both interactions are exact duplicates.
    """

    if not interactions_share_context(
        first,
        second,
        include_pose=include_pose,
        include_model=include_model,
    ):
        return False

    return interactions_share_group_pair(
        first,
        second,
    )


def interactions_are_atomic_duplicates(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
    *,
    minimum_overlap: float = 0.5,
    include_pose: bool = True,
    include_model: bool = True,
) -> bool:
    """
    Return whether two interactions substantially overlap atomically.

    Both cationic and anionic atom sets must satisfy the overlap threshold.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.
    minimum_overlap
        Minimum overlap fraction for both charged sides.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    bool
        Whether the interactions are atomic duplicates.
    """

    normalized_overlap = safe_float(minimum_overlap)

    if (
        normalized_overlap is None
        or not 0.0 <= normalized_overlap <= 1.0
    ):
        raise SaltBridgeDetectionError(
            "minimum_overlap must be between 0.0 and 1.0."
        )

    if not interactions_share_context(
        first,
        second,
        include_pose=include_pose,
        include_model=include_model,
    ):
        return False

    if not interactions_share_residue_pair(first, second):
        return False

    cation_overlap, anion_overlap = (
        calculate_interaction_atomic_overlap(
            first,
            second,
        )
    )

    return (
        cation_overlap >= normalized_overlap
        and anion_overlap >= normalized_overlap
    )


def interactions_are_residue_duplicates(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
    *,
    distance_tolerance: float = 0.25,
    include_pose: bool = True,
    include_model: bool = True,
) -> bool:
    """
    Return whether interactions represent equivalent contacts for one residue
    pair.

    Residue-level duplicates require matching context and a sufficiently small
    difference between minimum atom distances.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.
    distance_tolerance
        Maximum allowed absolute distance difference.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    bool
        Whether the interactions are residue-level duplicates.
    """

    normalized_tolerance = safe_float(distance_tolerance)

    if normalized_tolerance is None or normalized_tolerance < 0.0:
        raise SaltBridgeDetectionError(
            "distance_tolerance must be finite and non-negative."
        )

    if not interactions_share_context(
        first,
        second,
        include_pose=include_pose,
        include_model=include_model,
    ):
        return False

    if not interactions_share_residue_pair(first, second):
        return False

    first_distance = safe_float(
        first.distance,
        default=math.inf,
    )

    second_distance = safe_float(
        second.distance,
        default=math.inf,
    )

    if (
        first_distance is None
        or second_distance is None
        or not math.isfinite(first_distance)
        or not math.isfinite(second_distance)
    ):
        return False

    return (
        abs(first_distance - second_distance)
        <= normalized_tolerance
    )


def interactions_are_duplicates(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    include_pose: bool = True,
    include_model: bool = True,
) -> bool:
    """
    Return whether two salt bridges should be considered duplicates.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.
    config
        Salt-bridge configuration.
    mode
        Optional explicit deduplication mode. Supported values are
        ``"exact"``, ``"atomic_overlap"``, and ``"residue_pair"``.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    bool
        Whether the interactions are duplicates.
    """

    resolved_config = resolve_config(config)

    configured_mode = getattr(
        resolved_config,
        "interaction_deduplication_mode",
        "atomic_overlap",
    )

    normalized_mode = normalize_text(
        mode if mode is not None else configured_mode,
        default="atomic_overlap",
        lowercase=True,
    )

    if normalized_mode == "exact":
        return interactions_are_exact_duplicates(
            first,
            second,
            include_pose=include_pose,
            include_model=include_model,
        )

    if normalized_mode == "atomic_overlap":
        minimum_overlap = safe_float(
            getattr(
                resolved_config,
                "interaction_overlap_threshold",
                0.5,
            ),
            default=0.5,
        )

        return interactions_are_atomic_duplicates(
            first,
            second,
            minimum_overlap=minimum_overlap or 0.5,
            include_pose=include_pose,
            include_model=include_model,
        )

    if normalized_mode == "residue_pair":
        distance_tolerance = safe_float(
            getattr(
                resolved_config,
                "deduplication_distance_tolerance",
                0.25,
            ),
            default=0.25,
        )

        return interactions_are_residue_duplicates(
            first,
            second,
            distance_tolerance=distance_tolerance or 0.0,
            include_pose=include_pose,
            include_model=include_model,
        )

    raise SaltBridgeDetectionError(
        f"Unsupported interaction deduplication mode: {normalized_mode!r}."
    )


# =============================================================================
# 11.4. INTERACTION QUALITY RANKING
# =============================================================================


_STRENGTH_PRIORITY = {
    STRENGTH_STRONG: 4,
    STRENGTH_MODERATE: 3,
    STRENGTH_WEAK: 2,
    STRENGTH_REJECTED: 1,
}


def interaction_strength_priority(
    interaction: SaltBridgeInteraction,
) -> int:
    """
    Return the ranking priority of an interaction strength.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    int
        Strength-priority value.
    """

    return _STRENGTH_PRIORITY.get(
        normalize_text(
            interaction.strength,
            lowercase=True,
        ),
        0,
    )


def interaction_recognition_confidence(
    interaction: SaltBridgeInteraction,
) -> float:
    """
    Return the joint recognition confidence of an interaction.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    float
        Joint confidence between 0.0 and 1.0.
    """

    return calculate_group_confidence_factor(
        interaction.cation,
        interaction.anion,
    )


def interaction_quality_key(
    interaction: SaltBridgeInteraction,
) -> Tuple[Any, ...]:
    """
    Build a sortable interaction-quality key.

    Better interactions produce lexicographically larger values. Ranking
    considers:

    1. geometric validity;
    2. score;
    3. strength;
    4. recognition confidence;
    5. atomic contact count;
    6. shorter minimum distance;
    7. shorter center distance.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Tuple[Any, ...]
        Quality-ranking key.
    """

    valid_priority = (
        1
        if interaction.geometry.valid
        else 0
    )

    score = safe_float(
        interaction.score,
        default=0.0,
    )

    contact_count = safe_int(
        interaction.geometry.contact_count,
        default=0,
    )

    minimum_distance = safe_float(
        interaction.distance,
        default=math.inf,
    )

    center_distance = safe_float(
        interaction.center_distance,
        default=math.inf,
    )

    return (
        valid_priority,
        score or 0.0,
        interaction_strength_priority(interaction),
        interaction_recognition_confidence(interaction),
        contact_count or 0,
        -(minimum_distance or math.inf),
        -(center_distance or math.inf),
    )


def select_preferred_interaction(
    first: SaltBridgeInteraction,
    second: SaltBridgeInteraction,
) -> SaltBridgeInteraction:
    """
    Select the preferred representation of two duplicate interactions.

    Parameters
    ----------
    first
        First interaction.
    second
        Second interaction.

    Returns
    -------
    SaltBridgeInteraction
        Preferred interaction.
    """

    first_key = interaction_quality_key(first)
    second_key = interaction_quality_key(second)

    if second_key > first_key:
        return second

    return first


def merge_duplicate_interaction_metadata(
    preferred: SaltBridgeInteraction,
    discarded: SaltBridgeInteraction,
) -> SaltBridgeInteraction:
    """
    Record duplicate provenance in the retained interaction.

    Parameters
    ----------
    preferred
        Retained interaction.
    discarded
        Duplicate interaction being removed.

    Returns
    -------
    SaltBridgeInteraction
        Retained interaction with updated metadata.
    """

    duplicate_ids = preferred.metadata.setdefault(
        "merged_duplicate_ids",
        [],
    )

    discarded_identifier = (
        discarded.interaction_id
        or make_interaction_identifier(
            discarded.cation,
            discarded.anion,
            pose_id=discarded.pose_id,
            model_id=discarded.model_id,
        )
    )

    if discarded_identifier not in duplicate_ids:
        duplicate_ids.append(discarded_identifier)

    preferred.metadata["duplicate_count"] = (
        len(duplicate_ids)
    )

    preferred.metadata["deduplicated"] = True

    return preferred


# =============================================================================
# 11.5. LINEAR KEY-BASED DEDUPLICATION
# =============================================================================


def deduplicate_interactions_by_key(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    key_mode: str = "group_pair",
    include_pose: bool = True,
    include_model: bool = True,
    merge_metadata: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Deduplicate interactions using an exact hashable identity key.

    This method has approximately linear complexity and should be preferred
    when exact group-, atom-, or residue-pair identity is sufficient.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    key_mode
        Identity-key mode.
    include_pose
        Whether pose identifiers should be part of the identity.
    include_model
        Whether model identifiers should be part of the identity.
    merge_metadata
        Whether duplicate provenance should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Deduplicated interactions.
    """

    retained_by_key: Dict[
        Tuple[Any, ...],
        SaltBridgeInteraction,
    ] = {}

    key_order: List[Tuple[Any, ...]] = []

    for interaction in interactions:
        if not isinstance(interaction, SaltBridgeInteraction):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeInteraction instances."
            )

        identity_key = interaction_identity_key(
            interaction,
            mode=key_mode,
            include_pose=include_pose,
            include_model=include_model,
        )

        existing_interaction = retained_by_key.get(
            identity_key
        )

        if existing_interaction is None:
            retained_by_key[identity_key] = interaction
            key_order.append(identity_key)
            continue

        preferred_interaction = select_preferred_interaction(
            existing_interaction,
            interaction,
        )

        discarded_interaction = (
            interaction
            if preferred_interaction is existing_interaction
            else existing_interaction
        )

        if merge_metadata:
            merge_duplicate_interaction_metadata(
                preferred_interaction,
                discarded_interaction,
            )

        retained_by_key[identity_key] = (
            preferred_interaction
        )

    return [
        retained_by_key[identity_key]
        for identity_key in key_order
    ]


# =============================================================================
# 11.6. OVERLAP-BASED DEDUPLICATION
# =============================================================================


def deduplicate_interactions_by_overlap(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    include_pose: bool = True,
    include_model: bool = True,
    merge_metadata: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Deduplicate interactions using pairwise duplicate evaluation.

    This method supports partial atomic overlap and residue-level comparison.
    It is more flexible than exact key-based deduplication but may have
    quadratic worst-case complexity.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.
    mode
        Optional duplicate-comparison mode.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.
    merge_metadata
        Whether duplicate provenance should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Deduplicated interactions.
    """

    resolved_config = resolve_config(config)
    retained_interactions: List[SaltBridgeInteraction] = []

    for candidate in interactions:
        if not isinstance(candidate, SaltBridgeInteraction):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeInteraction instances."
            )

        duplicate_index: Optional[int] = None

        for index, retained in enumerate(
            retained_interactions
        ):
            if interactions_are_duplicates(
                candidate,
                retained,
                resolved_config,
                mode=mode,
                include_pose=include_pose,
                include_model=include_model,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            retained_interactions.append(candidate)
            continue

        retained = retained_interactions[
            duplicate_index
        ]

        preferred = select_preferred_interaction(
            retained,
            candidate,
        )

        discarded = (
            candidate
            if preferred is retained
            else retained
        )

        if merge_metadata:
            merge_duplicate_interaction_metadata(
                preferred,
                discarded,
            )

        retained_interactions[
            duplicate_index
        ] = preferred

    return retained_interactions


# =============================================================================
# 11.7. PUBLIC INTERACTION DEDUPLICATION API
# =============================================================================


def deduplicate_salt_bridge_interactions(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    include_pose: bool = True,
    include_model: bool = True,
    merge_metadata: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Deduplicate a salt-bridge interaction collection.

    The function automatically selects exact key-based or overlap-based
    processing according to the requested mode.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.
    mode
        Optional deduplication mode. Supported values are ``"exact"``,
        ``"group_pair"``, ``"atom_pair"``, ``"atomic_overlap"``, and
        ``"residue_pair"``.
    include_pose
        Whether pose identifiers should isolate duplicate groups.
    include_model
        Whether model identifiers should isolate duplicate groups.
    merge_metadata
        Whether removed duplicate identifiers should be stored.

    Returns
    -------
    List[SaltBridgeInteraction]
        Deduplicated interactions.
    """

    resolved_config = resolve_config(config)
    interaction_list = list(interactions)

    if not getattr(
        resolved_config,
        "deduplicate_interactions",
        True,
    ):
        return interaction_list

    configured_mode = getattr(
        resolved_config,
        "interaction_deduplication_mode",
        "atomic_overlap",
    )

    normalized_mode = normalize_text(
        mode if mode is not None else configured_mode,
        default="atomic_overlap",
        lowercase=True,
    )

    if normalized_mode == "exact":
        normalized_mode = "group_pair"

    if normalized_mode in {
        "group_pair",
        "atom_pair",
    }:
        return deduplicate_interactions_by_key(
            interaction_list,
            key_mode=normalized_mode,
            include_pose=include_pose,
            include_model=include_model,
            merge_metadata=merge_metadata,
        )

    if normalized_mode in {
        "atomic_overlap",
        "residue_pair",
    }:
        return deduplicate_interactions_by_overlap(
            interaction_list,
            resolved_config,
            mode=normalized_mode,
            include_pose=include_pose,
            include_model=include_model,
            merge_metadata=merge_metadata,
        )

    raise SaltBridgeDetectionError(
        f"Unsupported salt-bridge deduplication mode: "
        f"{normalized_mode!r}."
    )


# =============================================================================
# 11.8. INTERACTION IDENTIFIER REFRESH
# =============================================================================


def refresh_interaction_identifiers(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    preserve_existing: bool = False,
) -> List[SaltBridgeInteraction]:
    """
    Assign deterministic sequential identifiers after deduplication.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    preserve_existing
        Whether existing non-empty identifiers should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Same interaction objects with refreshed identifiers.
    """

    interaction_list = list(interactions)

    for index, interaction in enumerate(
        interaction_list,
        start=1,
    ):
        if (
            preserve_existing
            and interaction.interaction_id
        ):
            continue

        interaction.interaction_id = (
            make_interaction_identifier(
                interaction.cation,
                interaction.anion,
                pose_id=interaction.pose_id,
                model_id=interaction.model_id,
                index=index,
            )
        )

    return interaction_list


# =============================================================================
# 11.9. RESULT-LEVEL DEDUPLICATION
# =============================================================================


def deduplicate_salt_bridge_result(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    in_place: bool = True,
    include_pose: bool = True,
    include_model: bool = True,
    merge_metadata: bool = True,
    refresh_identifiers: bool = True,
) -> SaltBridgeResult:
    """
    Deduplicate all interactions stored in a SaltBridgeResult.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.
    mode
        Optional deduplication mode.
    in_place
        Whether the original result should be modified.
    include_pose
        Whether pose identifiers should isolate duplicates.
    include_model
        Whether model identifiers should isolate duplicates.
    merge_metadata
        Whether duplicate provenance should be retained.
    refresh_identifiers
        Whether identifiers should be regenerated after deduplication.

    Returns
    -------
    SaltBridgeResult
        Result containing deduplicated interactions.
    """

    resolved_config = resolve_config(config)

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    target_result = result

    if not in_place:
        target_result = SaltBridgeResult(
            interactions=list(result.interactions),
            cationic_groups=list(result.cationic_groups),
            anionic_groups=list(result.anionic_groups),
            statistics=dict(result.statistics),
            warnings=list(result.warnings),
            pose_id=result.pose_id,
            model_id=result.model_id,
            metadata=dict(result.metadata),
        )

    original_count = len(
        target_result.interactions
    )

    deduplicated_interactions = (
        deduplicate_salt_bridge_interactions(
            target_result.interactions,
            resolved_config,
            mode=mode,
            include_pose=include_pose,
            include_model=include_model,
            merge_metadata=merge_metadata,
        )
    )

    if refresh_identifiers:
        deduplicated_interactions = (
            refresh_interaction_identifiers(
                deduplicated_interactions,
                preserve_existing=False,
            )
        )

    target_result.interactions = (
        deduplicated_interactions
    )

    final_count = len(
        deduplicated_interactions
    )

    removed_count = max(
        0,
        original_count - final_count,
    )

    target_result.metadata[
        "deduplication_completed"
    ] = True

    target_result.metadata[
        "pre_deduplication_interaction_count"
    ] = original_count

    target_result.metadata[
        "deduplicated_interaction_count"
    ] = final_count

    target_result.metadata[
        "removed_duplicate_count"
    ] = removed_count

    target_result.metadata[
        "interaction_deduplication_mode"
    ] = normalize_text(
        mode
        if mode is not None
        else getattr(
            resolved_config,
            "interaction_deduplication_mode",
            "atomic_overlap",
        ),
        lowercase=True,
    )

    return target_result


# =============================================================================
# 11.10. DUPLICATE GROUP INSPECTION
# =============================================================================


def group_duplicate_interactions(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    include_pose: bool = True,
    include_model: bool = True,
) -> List[List[SaltBridgeInteraction]]:
    """
    Group interactions into duplicate-equivalence collections.

    This function is intended for diagnostics and self-tests. It does not
    remove any interaction.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.
    mode
        Optional duplicate-comparison mode.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    List[List[SaltBridgeInteraction]]
        Duplicate-equivalence groups.
    """

    resolved_config = resolve_config(config)
    duplicate_groups: List[
        List[SaltBridgeInteraction]
    ] = []

    for interaction in interactions:
        assigned = False

        for duplicate_group in duplicate_groups:
            representative = duplicate_group[0]

            if interactions_are_duplicates(
                interaction,
                representative,
                resolved_config,
                mode=mode,
                include_pose=include_pose,
                include_model=include_model,
            ):
                duplicate_group.append(interaction)
                assigned = True
                break

        if not assigned:
            duplicate_groups.append(
                [interaction]
            )

    return duplicate_groups


def find_duplicate_salt_bridges(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
    *,
    mode: Optional[str] = None,
    include_pose: bool = True,
    include_model: bool = True,
) -> List[List[SaltBridgeInteraction]]:
    """
    Return only interaction groups containing duplicates.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.
    mode
        Optional deduplication mode.
    include_pose
        Whether pose identifiers must match.
    include_model
        Whether model identifiers must match.

    Returns
    -------
    List[List[SaltBridgeInteraction]]
        Groups containing at least two interactions.
    """

    return [
        duplicate_group
        for duplicate_group in group_duplicate_interactions(
            interactions,
            config,
            mode=mode,
            include_pose=include_pose,
            include_model=include_model,
        )
        if len(duplicate_group) > 1
    ]


# =============================================================================
# 11.11. COMPLETE PIPELINE THROUGH DEDUPLICATION
# =============================================================================


def analyze_and_deduplicate_salt_bridges(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
    deduplication_mode: Optional[str] = None,
) -> SaltBridgeResult:
    """
    Recognize, detect, classify, score, and deduplicate salt bridges.

    This function combines Sections 7 through 11. Grouping, statistics,
    DockModel integration, multipose handling, serialization, and ChimeraX
    compatibility remain separate.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.
    deduplication_mode
        Optional interaction deduplication mode.

    Returns
    -------
    SaltBridgeResult
        Classified, scored, and deduplicated result.
    """

    resolved_config = resolve_config(config)

    result = analyze_salt_bridges(
        source,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
    )

    return deduplicate_salt_bridge_result(
        result,
        resolved_config,
        mode=deduplication_mode,
        in_place=True,
        include_pose=True,
        include_model=True,
        merge_metadata=not resolved_config.compact_results,
        refresh_identifiers=True,
    )


# =============================================================================
# 12. GROUPING
# =============================================================================


# =============================================================================
# 12.1. GROUPING KEY NORMALIZATION
# =============================================================================


def normalize_grouping_identifier(
    value: Any,
    *,
    fallback: str = "unknown",
) -> str:
    """
    Normalize a value used as a grouping identifier.

    Parameters
    ----------
    value
        Identifier-like value.
    fallback
        Value returned when the identifier is unavailable.

    Returns
    -------
    str
        Normalized grouping identifier.
    """

    if value is None:
        return fallback

    normalized_value = normalize_text(
        value,
        default=fallback,
    )

    return normalized_value or fallback


def charged_group_grouping_key(
    group: ChargedGroup,
    *,
    include_group_type: bool = True,
    include_polarity: bool = True,
) -> Tuple[Any, ...]:
    """
    Build a stable grouping key for a charged group.

    Parameters
    ----------
    group
        Charged group.
    include_group_type
        Whether the chemical group type should be included.
    include_polarity
        Whether group polarity should be included.

    Returns
    -------
    Tuple[Any, ...]
        Hashable charged-group key.
    """

    if not isinstance(group, ChargedGroup):
        raise SaltBridgeDetectionError(
            "group must be a ChargedGroup instance."
        )

    key_parts: List[Any] = [
        residue_identity(group.residue),
        tuple(
            sorted(
                (
                    repr(charged_atom_identity(charged_atom)),
                    charged_atom_identity(charged_atom),
                )
                for charged_atom in group.atoms
            )
        ),
    ]

    if include_group_type:
        key_parts.append(group.group_type)

    if include_polarity:
        key_parts.append(group.polarity)

    return tuple(key_parts)


def interaction_residue_grouping_key(
    interaction: SaltBridgeInteraction,
    *,
    directional: bool = True,
) -> Tuple[Any, ...]:
    """
    Build a residue-pair grouping key for an interaction.

    Salt bridges are intrinsically directional because one side is cationic
    and the other is anionic. When ``directional`` is disabled, both residue
    identities are sorted to produce an undirected residue-pair key.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    directional
        Whether cation-anion direction should be preserved.

    Returns
    -------
    Tuple[Any, ...]
        Hashable residue-pair key.
    """

    if not isinstance(interaction, SaltBridgeInteraction):
        raise SaltBridgeDetectionError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    cation_residue_key = residue_identity(
        interaction.cation.residue
    )

    anion_residue_key = residue_identity(
        interaction.anion.residue
    )

    if directional:
        return (
            "cation_to_anion",
            cation_residue_key,
            anion_residue_key,
        )

    ordered_residues = tuple(
        sorted(
            (
                (
                    repr(cation_residue_key),
                    cation_residue_key,
                ),
                (
                    repr(anion_residue_key),
                    anion_residue_key,
                ),
            )
        )
    )

    return (
        "undirected_residue_pair",
        ordered_residues,
    )


def interaction_group_pair_grouping_key(
    interaction: SaltBridgeInteraction,
) -> Tuple[Any, ...]:
    """
    Build a grouping key from the complete charged-group pair.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Tuple[Any, ...]
        Hashable charged-group-pair key.
    """

    return (
        "charged_group_pair",
        charged_group_grouping_key(
            interaction.cation
        ),
        charged_group_grouping_key(
            interaction.anion
        ),
    )


def interaction_chain_pair_grouping_key(
    interaction: SaltBridgeInteraction,
    *,
    directional: bool = True,
) -> Tuple[str, str]:
    """
    Build a chain-pair grouping key.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    directional
        Whether cation-anion chain direction should be preserved.

    Returns
    -------
    Tuple[str, str]
        Cation and anion chain identifiers.
    """

    cation_chain = normalize_grouping_identifier(
        get_chain_id(interaction.cation.residue),
        fallback="unknown_chain",
    )

    anion_chain = normalize_grouping_identifier(
        get_chain_id(interaction.anion.residue),
        fallback="unknown_chain",
    )

    if directional:
        return cation_chain, anion_chain

    return tuple(
        sorted(
            (
                cation_chain,
                anion_chain,
            )
        )
    )


def interaction_pose_grouping_key(
    interaction: SaltBridgeInteraction,
) -> str:
    """
    Return a normalized pose grouping key.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    str
        Pose identifier.
    """

    return normalize_grouping_identifier(
        interaction.pose_id,
        fallback="unassigned_pose",
    )


def interaction_model_grouping_key(
    interaction: SaltBridgeInteraction,
) -> str:
    """
    Return a normalized model grouping key.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    str
        Model identifier.
    """

    return normalize_grouping_identifier(
        interaction.model_id,
        fallback="unassigned_model",
    )


# =============================================================================
# 12.2. GENERIC GROUPING UTILITIES
# =============================================================================


def group_interactions_by_key(
    interactions: Iterable[SaltBridgeInteraction],
    key_function: Callable[
        [SaltBridgeInteraction],
        Hashable,
    ],
) -> Dict[Hashable, List[SaltBridgeInteraction]]:
    """
    Group interactions using a custom key function.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    key_function
        Function returning a hashable grouping key.

    Returns
    -------
    Dict[Hashable, List[SaltBridgeInteraction]]
        Mapping from keys to interaction lists.
    """

    if not callable(key_function):
        raise SaltBridgeDetectionError(
            "key_function must be callable."
        )

    grouped_interactions: Dict[
        Hashable,
        List[SaltBridgeInteraction],
    ] = defaultdict(list)

    for interaction in interactions:
        if not isinstance(interaction, SaltBridgeInteraction):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeInteraction instances."
            )

        grouping_key = key_function(interaction)

        try:
            hash(grouping_key)
        except TypeError as error:
            raise SaltBridgeDetectionError(
                "Grouping keys must be hashable."
            ) from error

        grouped_interactions[grouping_key].append(
            interaction
        )

    return dict(grouped_interactions)


def sort_interaction_groups(
    grouped_interactions: Mapping[
        Hashable,
        Iterable[SaltBridgeInteraction],
    ],
    *,
    sort_interactions: bool = True,
) -> Dict[Hashable, List[SaltBridgeInteraction]]:
    """
    Sort interactions inside grouped collections.

    Parameters
    ----------
    grouped_interactions
        Mapping of grouping keys to interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Hashable, List[SaltBridgeInteraction]]
        Normalized grouped interaction mapping.
    """

    normalized_groups: Dict[
        Hashable,
        List[SaltBridgeInteraction],
    ] = {}

    for grouping_key, interactions in grouped_interactions.items():
        interaction_list = list(interactions)

        if sort_interactions:
            interaction_list = sort_salt_bridges_by_score(
                interaction_list,
                descending=True,
            )

        normalized_groups[grouping_key] = interaction_list

    return normalized_groups


def filter_interaction_groups_by_size(
    grouped_interactions: Mapping[
        Hashable,
        Iterable[SaltBridgeInteraction],
    ],
    *,
    minimum_size: int = 1,
    maximum_size: Optional[int] = None,
) -> Dict[Hashable, List[SaltBridgeInteraction]]:
    """
    Filter grouped interaction collections by group size.

    Parameters
    ----------
    grouped_interactions
        Mapping of grouping keys to interactions.
    minimum_size
        Minimum accepted group size.
    maximum_size
        Optional maximum accepted group size.

    Returns
    -------
    Dict[Hashable, List[SaltBridgeInteraction]]
        Filtered grouping mapping.
    """

    normalized_minimum = safe_int(minimum_size)

    if normalized_minimum is None or normalized_minimum < 1:
        raise SaltBridgeDetectionError(
            "minimum_size must be at least one."
        )

    normalized_maximum = None

    if maximum_size is not None:
        normalized_maximum = safe_int(maximum_size)

        if normalized_maximum is None or normalized_maximum < 1:
            raise SaltBridgeDetectionError(
                "maximum_size must be at least one."
            )

        if normalized_maximum < normalized_minimum:
            raise SaltBridgeDetectionError(
                "maximum_size cannot be smaller than minimum_size."
            )

    filtered_groups: Dict[
        Hashable,
        List[SaltBridgeInteraction],
    ] = {}

    for grouping_key, interactions in grouped_interactions.items():
        interaction_list = list(interactions)
        group_size = len(interaction_list)

        if group_size < normalized_minimum:
            continue

        if (
            normalized_maximum is not None
            and group_size > normalized_maximum
        ):
            continue

        filtered_groups[grouping_key] = interaction_list

    return filtered_groups


# =============================================================================
# 12.3. RESIDUE-LEVEL GROUPING
# =============================================================================


def group_salt_bridges_by_residue_pair(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    directional: bool = True,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[Any, ...],
    List[SaltBridgeInteraction],
]:
    """
    Group salt bridges by interacting residue pair.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    directional
        Whether cation-anion direction should be preserved.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[Any, ...], List[SaltBridgeInteraction]]
        Residue-pair groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        lambda interaction: interaction_residue_grouping_key(
            interaction,
            directional=directional,
        ),
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_cation_residue(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[Any, ...],
    List[SaltBridgeInteraction],
]:
    """
    Group interactions by cationic residue.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[Any, ...], List[SaltBridgeInteraction]]
        Cationic-residue groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        lambda interaction: residue_identity(
            interaction.cation.residue
        ),
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_anion_residue(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[Any, ...],
    List[SaltBridgeInteraction],
]:
    """
    Group interactions by anionic residue.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[Any, ...], List[SaltBridgeInteraction]]
        Anionic-residue groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        lambda interaction: residue_identity(
            interaction.anion.residue
        ),
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_any_residue(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[Any, ...],
    List[SaltBridgeInteraction],
]:
    """
    Group interactions by every participating residue.

    Each salt bridge is included once in the cationic residue group and once
    in the anionic residue group. Self-pairs, if present, are included only
    once.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[Any, ...], List[SaltBridgeInteraction]]
        Residue-to-interaction mapping.
    """

    grouped_interactions: Dict[
        Tuple[Any, ...],
        List[SaltBridgeInteraction],
    ] = defaultdict(list)

    for interaction in interactions:
        cation_residue_key = residue_identity(
            interaction.cation.residue
        )

        anion_residue_key = residue_identity(
            interaction.anion.residue
        )

        grouped_interactions[cation_residue_key].append(
            interaction
        )

        if anion_residue_key != cation_residue_key:
            grouped_interactions[anion_residue_key].append(
                interaction
            )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


# =============================================================================
# 12.4. CHARGED-GROUP AND CHEMICAL-TYPE GROUPING
# =============================================================================


def group_salt_bridges_by_charged_group_pair(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[Any, ...],
    List[SaltBridgeInteraction],
]:
    """
    Group salt bridges by complete cation-anion charged-group identity.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[Any, ...], List[SaltBridgeInteraction]]
        Charged-group-pair groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        interaction_group_pair_grouping_key,
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_group_type(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    directional: bool = True,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[str, str],
    List[SaltBridgeInteraction],
]:
    """
    Group salt bridges by cationic and anionic chemical group types.

    Examples include:

    - ammonium to carboxylate;
    - guanidinium to phosphate;
    - imidazolium to sulfonate.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    directional
        Whether cation-anion direction should be preserved.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[str, str], List[SaltBridgeInteraction]]
        Chemical-type groups.
    """

    def make_type_key(
        interaction: SaltBridgeInteraction,
    ) -> Tuple[str, str]:
        cation_type = normalize_grouping_identifier(
            interaction.cation.group_type,
            fallback="unknown_cation",
        )

        anion_type = normalize_grouping_identifier(
            interaction.anion.group_type,
            fallback="unknown_anion",
        )

        if directional:
            return cation_type, anion_type

        return tuple(
            sorted(
                (
                    cation_type,
                    anion_type,
                )
            )
        )

    grouped_interactions = group_interactions_by_key(
        interactions,
        make_type_key,
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_strength(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_rejected: bool = False,
    sort_interactions: bool = True,
) -> Dict[str, List[SaltBridgeInteraction]]:
    """
    Group salt bridges by strength classification.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_rejected
        Whether rejected interactions should be included.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[str, List[SaltBridgeInteraction]]
        Strength groups.
    """

    grouped_interactions: Dict[
        str,
        List[SaltBridgeInteraction],
    ] = defaultdict(list)

    for interaction in interactions:
        strength = normalize_grouping_identifier(
            interaction.strength,
            fallback=STRENGTH_REJECTED,
        ).lower()

        if (
            strength == STRENGTH_REJECTED
            and not include_rejected
        ):
            continue

        grouped_interactions[strength].append(
            interaction
        )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


# =============================================================================
# 12.5. CHAIN, POSE, AND MODEL GROUPING
# =============================================================================


def group_salt_bridges_by_chain_pair(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    directional: bool = True,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[str, str],
    List[SaltBridgeInteraction],
]:
    """
    Group salt bridges by cationic and anionic chain pair.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    directional
        Whether cation-anion direction should be preserved.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[str, str], List[SaltBridgeInteraction]]
        Chain-pair groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        lambda interaction: interaction_chain_pair_grouping_key(
            interaction,
            directional=directional,
        ),
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_pose(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[str, List[SaltBridgeInteraction]]:
    """
    Group salt bridges by docking-pose identifier.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[str, List[SaltBridgeInteraction]]
        Pose groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        interaction_pose_grouping_key,
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_model(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[str, List[SaltBridgeInteraction]]:
    """
    Group salt bridges by molecular-model identifier.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[str, List[SaltBridgeInteraction]]
        Model groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        interaction_model_grouping_key,
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


def group_salt_bridges_by_model_and_pose(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[
    Tuple[str, str],
    List[SaltBridgeInteraction],
]:
    """
    Group salt bridges by model and pose identifiers.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[Tuple[str, str], List[SaltBridgeInteraction]]
        Model-pose groups.
    """

    grouped_interactions = group_interactions_by_key(
        interactions,
        lambda interaction: (
            interaction_model_grouping_key(
                interaction
            ),
            interaction_pose_grouping_key(
                interaction
            ),
        ),
    )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


# =============================================================================
# 12.6. INTERFACIAL GROUPING
# =============================================================================


def interaction_is_intrachain(
    interaction: SaltBridgeInteraction,
) -> bool:
    """
    Return whether both residues belong to the same chain.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    bool
        Whether the interaction is intrachain.
    """

    cation_chain, anion_chain = (
        interaction_chain_pair_grouping_key(
            interaction,
            directional=True,
        )
    )

    return cation_chain == anion_chain


def interaction_is_interchain(
    interaction: SaltBridgeInteraction,
) -> bool:
    """
    Return whether the residues belong to different chains.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    bool
        Whether the interaction is interchain.
    """

    return not interaction_is_intrachain(
        interaction
    )


def group_salt_bridges_by_interface_type(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    sort_interactions: bool = True,
) -> Dict[str, List[SaltBridgeInteraction]]:
    """
    Group interactions as intrachain or interchain.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    sort_interactions
        Whether interactions should be sorted by score.

    Returns
    -------
    Dict[str, List[SaltBridgeInteraction]]
        Interface-type groups.
    """

    grouped_interactions: Dict[
        str,
        List[SaltBridgeInteraction],
    ] = defaultdict(list)

    for interaction in interactions:
        interface_type = (
            "intrachain"
            if interaction_is_intrachain(interaction)
            else "interchain"
        )

        grouped_interactions[interface_type].append(
            interaction
        )

    return sort_interaction_groups(
        grouped_interactions,
        sort_interactions=sort_interactions,
    )


# =============================================================================
# 12.7. GROUP SUMMARY GENERATION
# =============================================================================


def summarize_interaction_group(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    group_key: Optional[Hashable] = None,
) -> Dict[str, Any]:
    """
    Build a compact summary for one interaction group.

    Parameters
    ----------
    interactions
        Salt-bridge interactions in one group.
    group_key
        Optional grouping key.

    Returns
    -------
    Dict[str, Any]
        Group summary.
    """

    interaction_list = list(interactions)

    valid_interactions = [
        interaction
        for interaction in interaction_list
        if interaction.geometry.valid
    ]

    rejected_interactions = [
        interaction
        for interaction in interaction_list
        if not interaction.geometry.valid
    ]

    scores = [
        float(interaction.score)
        for interaction in valid_interactions
        if safe_float(interaction.score) is not None
    ]

    distances = [
        float(interaction.distance)
        for interaction in valid_interactions
        if safe_float(interaction.distance) is not None
    ]

    strength_counts = {
        STRENGTH_STRONG: 0,
        STRENGTH_MODERATE: 0,
        STRENGTH_WEAK: 0,
        STRENGTH_REJECTED: 0,
    }

    for interaction in interaction_list:
        strength = normalize_text(
            interaction.strength,
            default=STRENGTH_REJECTED,
            lowercase=True,
        )

        strength_counts.setdefault(
            strength,
            0,
        )

        strength_counts[strength] += 1

    best_interaction = get_best_salt_bridge(
        valid_interactions
    )

    participating_residues = unique_preserve_order(
        (
            residue
            for interaction in interaction_list
            for residue in (
                interaction.cation.residue,
                interaction.anion.residue,
            )
            if residue is not None
        ),
        key=residue_identity,
    )

    participating_chains = unique_preserve_order(
        (
            get_chain_id(residue)
            for residue in participating_residues
        ),
    )

    return {
        "group_key": group_key,
        "interaction_count": len(interaction_list),
        "valid_interaction_count": len(valid_interactions),
        "rejected_interaction_count": len(rejected_interactions),
        "total_score": sum(scores) if scores else 0.0,
        "mean_score": (
            statistics.fmean(scores)
            if scores
            else 0.0
        ),
        "minimum_distance": (
            min(distances)
            if distances
            else None
        ),
        "maximum_distance": (
            max(distances)
            if distances
            else None
        ),
        "mean_distance": (
            statistics.fmean(distances)
            if distances
            else None
        ),
        "strength_counts": strength_counts,
        "residue_count": len(participating_residues),
        "chain_count": len(participating_chains),
        "best_interaction_id": (
            best_interaction.interaction_id
            if best_interaction is not None
            else None
        ),
        "best_score": (
            best_interaction.score
            if best_interaction is not None
            else 0.0
        ),
    }


def summarize_interaction_groups(
    grouped_interactions: Mapping[
        Hashable,
        Iterable[SaltBridgeInteraction],
    ],
) -> Dict[Hashable, Dict[str, Any]]:
    """
    Summarize all groups in a grouped-interaction mapping.

    Parameters
    ----------
    grouped_interactions
        Grouped salt-bridge interactions.

    Returns
    -------
    Dict[Hashable, Dict[str, Any]]
        Group summaries.
    """

    return {
        grouping_key: summarize_interaction_group(
            interactions,
            group_key=grouping_key,
        )
        for grouping_key, interactions
        in grouped_interactions.items()
    }


# =============================================================================
# 12.8. RESIDUE HOTSPOT ANALYSIS
# =============================================================================


def calculate_residue_hotspot_score(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    count_weight: float = 1.0,
    score_weight: float = 1.0,
    strong_bonus: float = 0.5,
    moderate_bonus: float = 0.25,
) -> float:
    """
    Calculate a residue hotspot score.

    The score combines interaction count, cumulative interaction score, and
    optional bonuses for strong and moderate salt bridges.

    Parameters
    ----------
    interactions
        Interactions involving one residue.
    count_weight
        Weight applied to the number of valid interactions.
    score_weight
        Weight applied to cumulative interaction score.
    strong_bonus
        Additional value per strong interaction.
    moderate_bonus
        Additional value per moderate interaction.

    Returns
    -------
    float
        Residue hotspot score.
    """

    normalized_count_weight = safe_float(
        count_weight,
        default=1.0,
    )

    normalized_score_weight = safe_float(
        score_weight,
        default=1.0,
    )

    normalized_strong_bonus = safe_float(
        strong_bonus,
        default=0.5,
    )

    normalized_moderate_bonus = safe_float(
        moderate_bonus,
        default=0.25,
    )

    interaction_list = [
        interaction
        for interaction in interactions
        if interaction.geometry.valid
    ]

    total_score = sum(
        max(
            0.0,
            safe_float(
                interaction.score,
                default=0.0,
            ) or 0.0,
        )
        for interaction in interaction_list
    )

    strong_count = sum(
        1
        for interaction in interaction_list
        if interaction.strength == STRENGTH_STRONG
    )

    moderate_count = sum(
        1
        for interaction in interaction_list
        if interaction.strength == STRENGTH_MODERATE
    )

    return (
        len(interaction_list)
        * (normalized_count_weight or 0.0)
        + total_score
        * (normalized_score_weight or 0.0)
        + strong_count
        * (normalized_strong_bonus or 0.0)
        + moderate_count
        * (normalized_moderate_bonus or 0.0)
    )


def identify_residue_hotspots(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    minimum_interactions: int = 2,
    minimum_hotspot_score: float = 0.0,
    include_singletons: bool = False,
) -> List[Dict[str, Any]]:
    """
    Identify residues participating in multiple or high-scoring salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    minimum_interactions
        Minimum number of valid interactions required.
    minimum_hotspot_score
        Minimum accepted hotspot score.
    include_singletons
        Whether one-interaction residues may be retained.

    Returns
    -------
    List[Dict[str, Any]]
        Hotspot records sorted by decreasing hotspot score.
    """

    normalized_minimum_interactions = safe_int(
        minimum_interactions,
        default=2,
    )

    normalized_minimum_score = safe_float(
        minimum_hotspot_score,
        default=0.0,
    )

    if (
        normalized_minimum_interactions is None
        or normalized_minimum_interactions < 1
    ):
        raise SaltBridgeDetectionError(
            "minimum_interactions must be at least one."
        )

    residue_groups = group_salt_bridges_by_any_residue(
        interactions,
        sort_interactions=True,
    )

    hotspot_records: List[Dict[str, Any]] = []

    for residue_key, residue_interactions in residue_groups.items():
        valid_interactions = [
            interaction
            for interaction in residue_interactions
            if interaction.geometry.valid
        ]

        interaction_count = len(
            valid_interactions
        )

        if (
            not include_singletons
            and interaction_count
            < normalized_minimum_interactions
        ):
            continue

        hotspot_score = calculate_residue_hotspot_score(
            valid_interactions
        )

        if hotspot_score < (
            normalized_minimum_score or 0.0
        ):
            continue

        residue = None

        for interaction in valid_interactions:
            if (
                residue_identity(
                    interaction.cation.residue
                )
                == residue_key
            ):
                residue = interaction.cation.residue
                break

            if (
                residue_identity(
                    interaction.anion.residue
                )
                == residue_key
            ):
                residue = interaction.anion.residue
                break

        cationic_count = sum(
            1
            for interaction in valid_interactions
            if residue_identity(
                interaction.cation.residue
            ) == residue_key
        )

        anionic_count = sum(
            1
            for interaction in valid_interactions
            if residue_identity(
                interaction.anion.residue
            ) == residue_key
        )

        partner_residues = unique_preserve_order(
            (
                (
                    interaction.anion.residue
                    if residue_identity(
                        interaction.cation.residue
                    ) == residue_key
                    else interaction.cation.residue
                )
                for interaction in valid_interactions
            ),
            key=residue_identity,
        )

        best_interaction = get_best_salt_bridge(
            valid_interactions
        )

        hotspot_records.append(
            {
                "residue_key": residue_key,
                "residue": residue,
                "residue_label": make_residue_label(
                    residue,
                    fallback="unknown_residue",
                ),
                "interaction_count": interaction_count,
                "cationic_interaction_count": cationic_count,
                "anionic_interaction_count": anionic_count,
                "partner_count": len(partner_residues),
                "hotspot_score": hotspot_score,
                "total_interaction_score": sum(
                    interaction.score
                    for interaction in valid_interactions
                ),
                "strong_count": sum(
                    1
                    for interaction in valid_interactions
                    if interaction.strength
                    == STRENGTH_STRONG
                ),
                "moderate_count": sum(
                    1
                    for interaction in valid_interactions
                    if interaction.strength
                    == STRENGTH_MODERATE
                ),
                "weak_count": sum(
                    1
                    for interaction in valid_interactions
                    if interaction.strength
                    == STRENGTH_WEAK
                ),
                "minimum_distance": min(
                    interaction.distance
                    for interaction in valid_interactions
                ),
                "best_interaction_id": (
                    best_interaction.interaction_id
                    if best_interaction is not None
                    else None
                ),
                "interaction_ids": [
                    interaction.interaction_id
                    for interaction in valid_interactions
                ],
            }
        )

    hotspot_records.sort(
        key=lambda record: (
            -record["hotspot_score"],
            -record["interaction_count"],
            record["minimum_distance"],
            record["residue_label"],
        )
    )

    for rank, hotspot_record in enumerate(
        hotspot_records,
        start=1,
    ):
        hotspot_record["rank"] = rank

    return hotspot_records


# =============================================================================
# 12.9. COMPLETE GROUPING ASSEMBLY
# =============================================================================


def build_salt_bridge_groupings(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    Build the complete grouping collection for salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.

    Returns
    -------
    Dict[str, Any]
        Complete grouping dictionary.
    """

    resolved_config = resolve_config(config)
    interaction_list = list(interactions)

    include_rejected = bool(
        getattr(
            resolved_config,
            "include_rejected_in_grouping",
            False,
        )
    )

    grouping_interactions = (
        interaction_list
        if include_rejected
        else [
            interaction
            for interaction in interaction_list
            if interaction.geometry.valid
        ]
    )

    residue_pairs = group_salt_bridges_by_residue_pair(
        grouping_interactions,
        directional=True,
    )

    cation_residues = group_salt_bridges_by_cation_residue(
        grouping_interactions
    )

    anion_residues = group_salt_bridges_by_anion_residue(
        grouping_interactions
    )

    all_residues = group_salt_bridges_by_any_residue(
        grouping_interactions
    )

    charged_group_pairs = (
        group_salt_bridges_by_charged_group_pair(
            grouping_interactions
        )
    )

    group_types = group_salt_bridges_by_group_type(
        grouping_interactions
    )

    strengths = group_salt_bridges_by_strength(
        grouping_interactions,
        include_rejected=include_rejected,
    )

    chain_pairs = group_salt_bridges_by_chain_pair(
        grouping_interactions
    )

    interface_types = (
        group_salt_bridges_by_interface_type(
            grouping_interactions
        )
    )

    poses = group_salt_bridges_by_pose(
        grouping_interactions
    )

    models = group_salt_bridges_by_model(
        grouping_interactions
    )

    model_poses = group_salt_bridges_by_model_and_pose(
        grouping_interactions
    )

    minimum_hotspot_interactions = safe_int(
        getattr(
            resolved_config,
            "minimum_hotspot_interactions",
            2,
        ),
        default=2,
    )

    minimum_hotspot_score = safe_float(
        getattr(
            resolved_config,
            "minimum_hotspot_score",
            0.0,
        ),
        default=0.0,
    )

    hotspots = identify_residue_hotspots(
        grouping_interactions,
        minimum_interactions=(
            minimum_hotspot_interactions or 2
        ),
        minimum_hotspot_score=(
            minimum_hotspot_score or 0.0
        ),
        include_singletons=False,
    )

    return {
        "residue_pairs": residue_pairs,
        "cation_residues": cation_residues,
        "anion_residues": anion_residues,
        "all_residues": all_residues,
        "charged_group_pairs": charged_group_pairs,
        "group_types": group_types,
        "strengths": strengths,
        "chain_pairs": chain_pairs,
        "interface_types": interface_types,
        "poses": poses,
        "models": models,
        "model_poses": model_poses,
        "hotspots": hotspots,
        "summaries": {
            "residue_pairs": summarize_interaction_groups(
                residue_pairs
            ),
            "cation_residues": summarize_interaction_groups(
                cation_residues
            ),
            "anion_residues": summarize_interaction_groups(
                anion_residues
            ),
            "all_residues": summarize_interaction_groups(
                all_residues
            ),
            "charged_group_pairs": summarize_interaction_groups(
                charged_group_pairs
            ),
            "group_types": summarize_interaction_groups(
                group_types
            ),
            "strengths": summarize_interaction_groups(
                strengths
            ),
            "chain_pairs": summarize_interaction_groups(
                chain_pairs
            ),
            "interface_types": summarize_interaction_groups(
                interface_types
            ),
            "poses": summarize_interaction_groups(
                poses
            ),
            "models": summarize_interaction_groups(
                models
            ),
            "model_poses": summarize_interaction_groups(
                model_poses
            ),
        },
        "metadata": {
            "input_interaction_count": len(
                interaction_list
            ),
            "grouped_interaction_count": len(
                grouping_interactions
            ),
            "include_rejected": include_rejected,
            "residue_pair_group_count": len(
                residue_pairs
            ),
            "residue_group_count": len(
                all_residues
            ),
            "charged_group_pair_count": len(
                charged_group_pairs
            ),
            "group_type_count": len(
                group_types
            ),
            "chain_pair_count": len(
                chain_pairs
            ),
            "pose_count": len(poses),
            "model_count": len(models),
            "hotspot_count": len(hotspots),
        },
    }


# =============================================================================
# 12.10. RESULT-LEVEL GROUPING
# =============================================================================


def group_salt_bridge_result(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    in_place: bool = True,
    store_full_groups: Optional[bool] = None,
) -> SaltBridgeResult:
    """
    Build and attach grouping information to a SaltBridgeResult.

    Full interaction-group mappings may be stored in result metadata for
    immediate use. When compact storage is requested, only group summaries,
    hotspots, and grouping metadata are retained.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.
    in_place
        Whether the original result should be modified.
    store_full_groups
        Whether full grouped interaction mappings should be stored.

    Returns
    -------
    SaltBridgeResult
        Result with attached grouping data.
    """

    resolved_config = resolve_config(config)

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    target_result = result

    if not in_place:
        target_result = SaltBridgeResult(
            interactions=list(result.interactions),
            cationic_groups=list(
                result.cationic_groups
            ),
            anionic_groups=list(
                result.anionic_groups
            ),
            statistics=dict(result.statistics),
            warnings=list(result.warnings),
            pose_id=result.pose_id,
            model_id=result.model_id,
            metadata=dict(result.metadata),
        )

    grouping_data = build_salt_bridge_groupings(
        target_result.interactions,
        resolved_config,
    )

    if store_full_groups is None:
        store_full_groups = not bool(
            getattr(
                resolved_config,
                "compact_results",
                False,
            )
        )

    target_result.metadata[
        "grouping_completed"
    ] = True

    target_result.metadata[
        "grouping_metadata"
    ] = grouping_data["metadata"]

    target_result.metadata[
        "group_summaries"
    ] = grouping_data["summaries"]

    target_result.metadata[
        "hotspots"
    ] = grouping_data["hotspots"]

    if store_full_groups:
        target_result.metadata[
            "groups"
        ] = {
            key: value
            for key, value in grouping_data.items()
            if key not in {
                "summaries",
                "hotspots",
                "metadata",
            }
        }

    else:
        target_result.metadata.pop(
            "groups",
            None,
        )

    return target_result


# =============================================================================
# 12.11. GROUP ACCESS HELPERS
# =============================================================================


def get_result_groupings(
    result: SaltBridgeResult,
) -> Mapping[str, Any]:
    """
    Return full grouped interactions stored in a result.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    Mapping[str, Any]
        Stored grouping mapping.

    Raises
    ------
    SaltBridgeDetectionError
        If grouping data is unavailable.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    groupings = result.metadata.get(
        "groups"
    )

    if groupings is None:
        raise SaltBridgeDetectionError(
            "Full grouping data is not stored in this result."
        )

    return groupings


def get_result_group_summaries(
    result: SaltBridgeResult,
) -> Mapping[str, Any]:
    """
    Return grouping summaries stored in a result.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    Mapping[str, Any]
        Group summary mapping.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    return result.metadata.get(
        "group_summaries",
        {},
    )


def get_result_hotspots(
    result: SaltBridgeResult,
) -> List[Dict[str, Any]]:
    """
    Return residue hotspots stored in a result.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    List[Dict[str, Any]]
        Hotspot records.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    return list(
        result.metadata.get(
            "hotspots",
            [],
        )
    )


# =============================================================================
# 12.12. COMPLETE PIPELINE THROUGH GROUPING
# =============================================================================


def analyze_grouped_salt_bridges(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
    deduplication_mode: Optional[str] = None,
    store_full_groups: Optional[bool] = None,
) -> SaltBridgeResult:
    """
    Execute salt-bridge analysis through the grouping stage.

    The workflow includes:

    1. charged-group recognition;
    2. central detection;
    3. strength classification;
    4. scoring;
    5. interaction deduplication;
    6. grouping;
    7. hotspot identification.

    Statistics, DockModel integration, multipose orchestration,
    serialization, and ChimeraX compatibility remain separate.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.
    deduplication_mode
        Optional interaction deduplication mode.
    store_full_groups
        Whether full interaction groups should be stored.

    Returns
    -------
    SaltBridgeResult
        Classified, scored, deduplicated, and grouped result.
    """

    resolved_config = resolve_config(config)

    result = analyze_and_deduplicate_salt_bridges(
        source,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
        deduplication_mode=deduplication_mode,
    )

    return group_salt_bridge_result(
        result,
        resolved_config,
        in_place=True,
        store_full_groups=store_full_groups,
    )


# =============================================================================
# 13. STATISTICS AND SUMMARIES
# =============================================================================


# =============================================================================
# 13.1. NUMERIC STATISTICS UTILITIES
# =============================================================================


def calculate_numeric_statistics(
    values: Iterable[Any],
    *,
    ignore_invalid: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Calculate descriptive statistics for a numeric collection.

    Parameters
    ----------
    values
        Numeric or numeric-like values.
    ignore_invalid
        Whether invalid and non-finite values should be ignored.

    Returns
    -------
    Dict[str, Optional[float]]
        Count, sum, mean, median, minimum, maximum, standard deviation,
        variance, and quartiles.
    """

    numeric_values: List[float] = []

    for value in values:
        normalized_value = safe_float(value)

        if (
            normalized_value is None
            or not math.isfinite(normalized_value)
        ):
            if ignore_invalid:
                continue

            raise SaltBridgeDetectionError(
                "Numeric statistics require finite values."
            )

        numeric_values.append(normalized_value)

    value_count = len(numeric_values)

    if value_count == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
            "variance": None,
            "first_quartile": None,
            "third_quartile": None,
            "interquartile_range": None,
        }

    sorted_values = sorted(numeric_values)

    mean_value = statistics.fmean(
        sorted_values
    )

    median_value = statistics.median(
        sorted_values
    )

    variance_value = (
        statistics.pvariance(sorted_values)
        if value_count > 1
        else 0.0
    )

    standard_deviation = math.sqrt(
        variance_value
    )

    if value_count == 1:
        first_quartile = sorted_values[0]
        third_quartile = sorted_values[0]

    else:
        quartiles = statistics.quantiles(
            sorted_values,
            n=4,
            method="inclusive",
        )

        first_quartile = quartiles[0]
        third_quartile = quartiles[2]

    return {
        "count": value_count,
        "sum": sum(sorted_values),
        "mean": mean_value,
        "median": median_value,
        "minimum": sorted_values[0],
        "maximum": sorted_values[-1],
        "standard_deviation": standard_deviation,
        "variance": variance_value,
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "interquartile_range": (
            third_quartile - first_quartile
        ),
    }


def calculate_percentage(
    count: int,
    total: int,
) -> float:
    """
    Calculate a percentage while safely handling zero totals.

    Parameters
    ----------
    count
        Partial count.
    total
        Total count.

    Returns
    -------
    float
        Percentage between 0.0 and 100.0.
    """

    normalized_count = safe_int(
        count,
        default=0,
    )

    normalized_total = safe_int(
        total,
        default=0,
    )

    if (
        normalized_total is None
        or normalized_total <= 0
    ):
        return 0.0

    return (
        max(0, normalized_count or 0)
        / normalized_total
        * 100.0
    )


def normalize_count_distribution(
    counts: Mapping[Any, int],
) -> Dict[Any, Dict[str, Union[int, float]]]:
    """
    Convert raw counts into count-and-percentage records.

    Parameters
    ----------
    counts
        Mapping from category to count.

    Returns
    -------
    Dict[Any, Dict[str, Union[int, float]]]
        Category count and percentage records.
    """

    total_count = sum(
        max(
            0,
            safe_int(count, default=0) or 0,
        )
        for count in counts.values()
    )

    return {
        category: {
            "count": max(
                0,
                safe_int(count, default=0) or 0,
            ),
            "percentage": calculate_percentage(
                safe_int(count, default=0) or 0,
                total_count,
            ),
        }
        for category, count in counts.items()
    }


# =============================================================================
# 13.2. INTERACTION COLLECTION NORMALIZATION
# =============================================================================


def normalize_interaction_collection(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> List[SaltBridgeInteraction]:
    """
    Validate and normalize an interaction collection.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether geometrically invalid interactions should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Validated interaction list.
    """

    normalized_interactions: List[
        SaltBridgeInteraction
    ] = []

    for interaction in interactions:
        if not isinstance(
            interaction,
            SaltBridgeInteraction,
        ):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeInteraction instances."
            )

        if (
            not include_invalid
            and not interaction.geometry.valid
        ):
            continue

        normalized_interactions.append(
            interaction
        )

    return normalized_interactions


def get_scored_interactions(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_zero: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Return interactions containing valid finite scores.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_zero
        Whether zero-score interactions should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Scored interactions.
    """

    scored_interactions: List[
        SaltBridgeInteraction
    ] = []

    for interaction in interactions:
        score = safe_float(
            interaction.score
        )

        if score is None or not math.isfinite(score):
            continue

        if not include_zero and score <= 0.0:
            continue

        scored_interactions.append(
            interaction
        )

    return scored_interactions


# =============================================================================
# 13.3. GLOBAL INTERACTION COUNTS
# =============================================================================


def count_valid_salt_bridges(
    interactions: Iterable[SaltBridgeInteraction],
) -> int:
    """
    Count geometrically valid salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.

    Returns
    -------
    int
        Number of valid interactions.
    """

    return sum(
        1
        for interaction in interactions
        if interaction.geometry.valid
    )


def count_rejected_salt_bridges(
    interactions: Iterable[SaltBridgeInteraction],
) -> int:
    """
    Count geometrically rejected salt-bridge candidates.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.

    Returns
    -------
    int
        Number of rejected candidates.
    """

    return sum(
        1
        for interaction in interactions
        if not interaction.geometry.valid
    )


def count_atomic_contacts(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    valid_only: bool = True,
) -> int:
    """
    Count all atomic contacts represented by salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    valid_only
        Whether invalid interactions should be ignored.

    Returns
    -------
    int
        Total atomic contact count.
    """

    total_contacts = 0

    for interaction in interactions:
        if (
            valid_only
            and not interaction.geometry.valid
        ):
            continue

        contact_count = safe_int(
            interaction.geometry.contact_count,
            default=0,
        )

        total_contacts += max(
            0,
            contact_count or 0,
        )

    return total_contacts


def calculate_interaction_count_statistics(
    interactions: Iterable[SaltBridgeInteraction],
) -> Dict[str, Any]:
    """
    Calculate global interaction-count statistics.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.

    Returns
    -------
    Dict[str, Any]
        Global counts and validity percentages.
    """

    interaction_list = list(interactions)

    total_count = len(interaction_list)

    valid_count = count_valid_salt_bridges(
        interaction_list
    )

    rejected_count = count_rejected_salt_bridges(
        interaction_list
    )

    atomic_contact_count = count_atomic_contacts(
        interaction_list,
        valid_only=True,
    )

    return {
        "total_interaction_count": total_count,
        "valid_interaction_count": valid_count,
        "rejected_interaction_count": rejected_count,
        "valid_percentage": calculate_percentage(
            valid_count,
            total_count,
        ),
        "rejected_percentage": calculate_percentage(
            rejected_count,
            total_count,
        ),
        "total_atomic_contact_count": (
            atomic_contact_count
        ),
        "mean_atomic_contacts_per_valid_interaction": (
            atomic_contact_count / valid_count
            if valid_count > 0
            else 0.0
        ),
    }


# =============================================================================
# 13.4. DISTANCE STATISTICS
# =============================================================================


def calculate_distance_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Calculate distance statistics for salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Dict[str, Optional[float]]]
        Minimum-atom, center, mean-contact, and maximum-contact statistics.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    minimum_atom_distances = [
        interaction.geometry.minimum_atom_distance
        for interaction in normalized_interactions
    ]

    center_distances = [
        interaction.geometry.center_distance
        for interaction in normalized_interactions
    ]

    mean_contact_distances = [
        interaction.geometry.mean_atom_distance
        for interaction in normalized_interactions
        if interaction.geometry.mean_atom_distance
        is not None
    ]

    maximum_contact_distances = [
        interaction.geometry.maximum_atom_distance
        for interaction in normalized_interactions
        if interaction.geometry.maximum_atom_distance
        is not None
    ]

    return {
        "minimum_atom_distance": (
            calculate_numeric_statistics(
                minimum_atom_distances
            )
        ),
        "center_distance": (
            calculate_numeric_statistics(
                center_distances
            )
        ),
        "mean_contact_distance": (
            calculate_numeric_statistics(
                mean_contact_distances
            )
        ),
        "maximum_contact_distance": (
            calculate_numeric_statistics(
                maximum_contact_distances
            )
        ),
    }


# =============================================================================
# 13.5. SCORE STATISTICS
# =============================================================================


def calculate_score_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
    include_zero: bool = True,
) -> Dict[str, Any]:
    """
    Calculate score statistics for salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.
    include_zero
        Whether zero scores should be included.

    Returns
    -------
    Dict[str, Any]
        Descriptive score statistics and best interaction information.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    scored_interactions = get_scored_interactions(
        normalized_interactions,
        include_zero=include_zero,
    )

    score_values = [
        interaction.score
        for interaction in scored_interactions
    ]

    score_statistics = (
        calculate_numeric_statistics(
            score_values
        )
    )

    best_interaction = get_best_salt_bridge(
        scored_interactions
    )

    score_statistics.update(
        {
            "scored_interaction_count": len(
                scored_interactions
            ),
            "best_interaction_id": (
                best_interaction.interaction_id
                if best_interaction is not None
                else None
            ),
            "best_interaction_score": (
                best_interaction.score
                if best_interaction is not None
                else None
            ),
            "best_interaction_distance": (
                best_interaction.distance
                if best_interaction is not None
                else None
            ),
            "best_interaction_strength": (
                best_interaction.strength
                if best_interaction is not None
                else None
            ),
        }
    )

    return score_statistics


# =============================================================================
# 13.6. STRENGTH DISTRIBUTION
# =============================================================================


def calculate_strength_distribution(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_rejected: bool = True,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Calculate the distribution of interaction strengths.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_rejected
        Whether rejected interactions should be included.

    Returns
    -------
    Dict[str, Dict[str, Union[int, float]]]
        Counts and percentages by strength.
    """

    strength_counts: Dict[str, int] = {
        STRENGTH_STRONG: 0,
        STRENGTH_MODERATE: 0,
        STRENGTH_WEAK: 0,
    }

    if include_rejected:
        strength_counts[
            STRENGTH_REJECTED
        ] = 0

    for interaction in interactions:
        strength = normalize_text(
            interaction.strength,
            default=STRENGTH_REJECTED,
            lowercase=True,
        )

        if (
            strength == STRENGTH_REJECTED
            and not include_rejected
        ):
            continue

        strength_counts.setdefault(
            strength,
            0,
        )

        strength_counts[strength] += 1

    return normalize_count_distribution(
        strength_counts
    )


# =============================================================================
# 13.7. CHEMICAL-TYPE DISTRIBUTION
# =============================================================================


def calculate_group_type_distribution(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    directional: bool = True,
    include_invalid: bool = False,
) -> Dict[
    Tuple[str, str],
    Dict[str, Union[int, float]],
]:
    """
    Calculate the distribution of cation-anion chemical group pairs.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    directional
        Whether cation-anion direction should be preserved.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[Tuple[str, str], Dict[str, Union[int, float]]]
        Counts and percentages by chemical group pair.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    group_type_counts: Dict[
        Tuple[str, str],
        int,
    ] = {}

    for interaction in normalized_interactions:
        cation_type = normalize_grouping_identifier(
            interaction.cation.group_type,
            fallback="unknown_cation",
        )

        anion_type = normalize_grouping_identifier(
            interaction.anion.group_type,
            fallback="unknown_anion",
        )

        if directional:
            group_type_key = (
                cation_type,
                anion_type,
            )

        else:
            group_type_key = tuple(
                sorted(
                    (
                        cation_type,
                        anion_type,
                    )
                )
            )

        group_type_counts[group_type_key] = (
            group_type_counts.get(
                group_type_key,
                0,
            )
            + 1
        )

    return normalize_count_distribution(
        group_type_counts
    )


def calculate_group_source_distribution(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Calculate distributions of charged-group recognition sources.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Any]
        Cationic, anionic, and paired source distributions.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    cation_source_counts: Dict[str, int] = {}
    anion_source_counts: Dict[str, int] = {}
    source_pair_counts: Dict[
        Tuple[str, str],
        int,
    ] = {}

    for interaction in normalized_interactions:
        cation_source = (
            normalize_grouping_identifier(
                interaction.cation.source,
                fallback="unknown",
            )
        )

        anion_source = (
            normalize_grouping_identifier(
                interaction.anion.source,
                fallback="unknown",
            )
        )

        cation_source_counts[cation_source] = (
            cation_source_counts.get(
                cation_source,
                0,
            )
            + 1
        )

        anion_source_counts[anion_source] = (
            anion_source_counts.get(
                anion_source,
                0,
            )
            + 1
        )

        source_pair_key = (
            cation_source,
            anion_source,
        )

        source_pair_counts[source_pair_key] = (
            source_pair_counts.get(
                source_pair_key,
                0,
            )
            + 1
        )

    return {
        "cation_sources": (
            normalize_count_distribution(
                cation_source_counts
            )
        ),
        "anion_sources": (
            normalize_count_distribution(
                anion_source_counts
            )
        ),
        "source_pairs": (
            normalize_count_distribution(
                source_pair_counts
            )
        ),
    }


# =============================================================================
# 13.8. RESIDUE STATISTICS
# =============================================================================


def collect_participating_residues(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    valid_only: bool = True,
) -> List[ResidueLike]:
    """
    Collect unique residues participating in salt bridges.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    valid_only
        Whether invalid interactions should be ignored.

    Returns
    -------
    List[ResidueLike]
        Unique participating residues.
    """

    residues: List[ResidueLike] = []

    for interaction in interactions:
        if (
            valid_only
            and not interaction.geometry.valid
        ):
            continue

        if interaction.cation.residue is not None:
            residues.append(
                interaction.cation.residue
            )

        if interaction.anion.residue is not None:
            residues.append(
                interaction.anion.residue
            )

    return unique_preserve_order(
        residues,
        key=residue_identity,
    )


def calculate_residue_participation_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Calculate residue participation statistics.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Any]
        Unique residue counts and residue-role distributions.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    cation_residue_groups = (
        group_salt_bridges_by_cation_residue(
            normalized_interactions
        )
    )

    anion_residue_groups = (
        group_salt_bridges_by_anion_residue(
            normalized_interactions
        )
    )

    all_residue_groups = (
        group_salt_bridges_by_any_residue(
            normalized_interactions
        )
    )

    participating_residues = (
        collect_participating_residues(
            normalized_interactions,
            valid_only=False,
        )
    )

    residue_interaction_counts = [
        len(residue_interactions)
        for residue_interactions
        in all_residue_groups.values()
    ]

    most_connected_residue_key = None
    most_connected_interaction_count = 0

    if all_residue_groups:
        (
            most_connected_residue_key,
            most_connected_residue_interactions,
        ) = max(
            all_residue_groups.items(),
            key=lambda item: (
                len(item[1]),
                sum(
                    interaction.score
                    for interaction in item[1]
                ),
            ),
        )

        most_connected_interaction_count = len(
            most_connected_residue_interactions
        )

    return {
        "unique_residue_count": len(
            participating_residues
        ),
        "unique_cationic_residue_count": len(
            cation_residue_groups
        ),
        "unique_anionic_residue_count": len(
            anion_residue_groups
        ),
        "residue_interaction_count_statistics": (
            calculate_numeric_statistics(
                residue_interaction_counts
            )
        ),
        "most_connected_residue_key": (
            most_connected_residue_key
        ),
        "most_connected_residue_interaction_count": (
            most_connected_interaction_count
        ),
    }


# =============================================================================
# 13.9. CHAIN AND INTERFACE STATISTICS
# =============================================================================


def calculate_interface_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Calculate intrachain, interchain, and chain-pair statistics.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Any]
        Interface and chain-pair statistics.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    interface_groups = (
        group_salt_bridges_by_interface_type(
            normalized_interactions
        )
    )

    chain_pair_groups = (
        group_salt_bridges_by_chain_pair(
            normalized_interactions
        )
    )

    intrachain_count = len(
        interface_groups.get(
            "intrachain",
            [],
        )
    )

    interchain_count = len(
        interface_groups.get(
            "interchain",
            [],
        )
    )

    total_count = len(
        normalized_interactions
    )

    chain_pair_counts = {
        chain_pair: len(
            chain_interactions
        )
        for chain_pair, chain_interactions
        in chain_pair_groups.items()
    }

    return {
        "intrachain_interaction_count": (
            intrachain_count
        ),
        "interchain_interaction_count": (
            interchain_count
        ),
        "intrachain_percentage": (
            calculate_percentage(
                intrachain_count,
                total_count,
            )
        ),
        "interchain_percentage": (
            calculate_percentage(
                interchain_count,
                total_count,
            )
        ),
        "unique_chain_pair_count": len(
            chain_pair_groups
        ),
        "chain_pair_distribution": (
            normalize_count_distribution(
                chain_pair_counts
            )
        ),
    }


# =============================================================================
# 13.10. POSE AND MODEL STATISTICS
# =============================================================================


def calculate_pose_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Calculate interaction statistics by docking pose.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Any]
        Pose counts, scores, and best-pose information.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    pose_groups = group_salt_bridges_by_pose(
        normalized_interactions
    )

    pose_summaries = summarize_interaction_groups(
        pose_groups
    )

    best_pose = None
    best_pose_score = None

    if pose_summaries:
        best_pose, best_summary = max(
            pose_summaries.items(),
            key=lambda item: (
                item[1]["total_score"],
                item[1]["interaction_count"],
                -(
                    item[1]["minimum_distance"]
                    if item[1]["minimum_distance"]
                    is not None
                    else math.inf
                ),
            ),
        )

        best_pose_score = best_summary[
            "total_score"
        ]

    interactions_per_pose = [
        summary["interaction_count"]
        for summary in pose_summaries.values()
    ]

    scores_per_pose = [
        summary["total_score"]
        for summary in pose_summaries.values()
    ]

    return {
        "pose_count": len(pose_groups),
        "interactions_per_pose": (
            calculate_numeric_statistics(
                interactions_per_pose
            )
        ),
        "scores_per_pose": (
            calculate_numeric_statistics(
                scores_per_pose
            )
        ),
        "best_pose_id": best_pose,
        "best_pose_score": best_pose_score,
        "pose_summaries": pose_summaries,
    }


def calculate_model_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Calculate interaction statistics by molecular model.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.

    Returns
    -------
    Dict[str, Any]
        Model counts, scores, and best-model information.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    model_groups = group_salt_bridges_by_model(
        normalized_interactions
    )

    model_summaries = summarize_interaction_groups(
        model_groups
    )

    best_model = None
    best_model_score = None

    if model_summaries:
        best_model, best_summary = max(
            model_summaries.items(),
            key=lambda item: (
                item[1]["total_score"],
                item[1]["interaction_count"],
                -(
                    item[1]["minimum_distance"]
                    if item[1]["minimum_distance"]
                    is not None
                    else math.inf
                ),
            ),
        )

        best_model_score = best_summary[
            "total_score"
        ]

    interactions_per_model = [
        summary["interaction_count"]
        for summary in model_summaries.values()
    ]

    scores_per_model = [
        summary["total_score"]
        for summary in model_summaries.values()
    ]

    return {
        "model_count": len(model_groups),
        "interactions_per_model": (
            calculate_numeric_statistics(
                interactions_per_model
            )
        ),
        "scores_per_model": (
            calculate_numeric_statistics(
                scores_per_model
            )
        ),
        "best_model_id": best_model,
        "best_model_score": best_model_score,
        "model_summaries": model_summaries,
    }


# =============================================================================
# 13.11. HOTSPOT STATISTICS
# =============================================================================


def calculate_hotspot_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate residue-hotspot statistics.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    config
        Salt-bridge configuration.

    Returns
    -------
    Dict[str, Any]
        Hotspot count, score distribution, and highest-ranked hotspot.
    """

    resolved_config = resolve_config(config)

    minimum_interactions = safe_int(
        getattr(
            resolved_config,
            "minimum_hotspot_interactions",
            2,
        ),
        default=2,
    )

    minimum_score = safe_float(
        getattr(
            resolved_config,
            "minimum_hotspot_score",
            0.0,
        ),
        default=0.0,
    )

    hotspots = identify_residue_hotspots(
        interactions,
        minimum_interactions=(
            minimum_interactions or 2
        ),
        minimum_hotspot_score=(
            minimum_score or 0.0
        ),
        include_singletons=False,
    )

    hotspot_scores = [
        hotspot["hotspot_score"]
        for hotspot in hotspots
    ]

    interaction_counts = [
        hotspot["interaction_count"]
        for hotspot in hotspots
    ]

    top_hotspot = (
        hotspots[0]
        if hotspots
        else None
    )

    return {
        "hotspot_count": len(hotspots),
        "hotspot_score_statistics": (
            calculate_numeric_statistics(
                hotspot_scores
            )
        ),
        "hotspot_interaction_count_statistics": (
            calculate_numeric_statistics(
                interaction_counts
            )
        ),
        "top_hotspot": top_hotspot,
        "hotspots": hotspots,
    }


# =============================================================================
# 13.12. RECOGNIZED GROUP STATISTICS
# =============================================================================


def calculate_recognized_group_statistics(
    cationic_groups: Iterable[ChargedGroup],
    anionic_groups: Iterable[ChargedGroup],
) -> Dict[str, Any]:
    """
    Calculate statistics for recognized charged groups.

    Parameters
    ----------
    cationic_groups
        Recognized positively charged groups.
    anionic_groups
        Recognized negatively charged groups.

    Returns
    -------
    Dict[str, Any]
        Group counts, source distributions, confidence statistics, and types.
    """

    cation_list = list(
        cationic_groups
    )

    anion_list = list(
        anionic_groups
    )

    cation_type_counts: Dict[str, int] = {}
    anion_type_counts: Dict[str, int] = {}

    cation_source_counts: Dict[str, int] = {}
    anion_source_counts: Dict[str, int] = {}

    for group in cation_list:
        group_type = normalize_grouping_identifier(
            group.group_type,
            fallback="unknown_cation",
        )

        group_source = normalize_grouping_identifier(
            group.source,
            fallback="unknown",
        )

        cation_type_counts[group_type] = (
            cation_type_counts.get(
                group_type,
                0,
            )
            + 1
        )

        cation_source_counts[group_source] = (
            cation_source_counts.get(
                group_source,
                0,
            )
            + 1
        )

    for group in anion_list:
        group_type = normalize_grouping_identifier(
            group.group_type,
            fallback="unknown_anion",
        )

        group_source = normalize_grouping_identifier(
            group.source,
            fallback="unknown",
        )

        anion_type_counts[group_type] = (
            anion_type_counts.get(
                group_type,
                0,
            )
            + 1
        )

        anion_source_counts[group_source] = (
            anion_source_counts.get(
                group_source,
                0,
            )
            + 1
        )

    cation_confidences = [
        group.confidence
        for group in cation_list
    ]

    anion_confidences = [
        group.confidence
        for group in anion_list
    ]

    return {
        "cationic_group_count": len(
            cation_list
        ),
        "anionic_group_count": len(
            anion_list
        ),
        "total_charged_group_count": (
            len(cation_list)
            + len(anion_list)
        ),
        "cation_type_distribution": (
            normalize_count_distribution(
                cation_type_counts
            )
        ),
        "anion_type_distribution": (
            normalize_count_distribution(
                anion_type_counts
            )
        ),
        "cation_source_distribution": (
            normalize_count_distribution(
                cation_source_counts
            )
        ),
        "anion_source_distribution": (
            normalize_count_distribution(
                anion_source_counts
            )
        ),
        "cation_confidence_statistics": (
            calculate_numeric_statistics(
                cation_confidences
            )
        ),
        "anion_confidence_statistics": (
            calculate_numeric_statistics(
                anion_confidences
            )
        ),
    }


# =============================================================================
# 13.13. COMPLETE STATISTICS ASSEMBLY
# =============================================================================


def calculate_salt_bridge_statistics(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    cationic_groups: Optional[
        Iterable[ChargedGroup]
    ] = None,
    anionic_groups: Optional[
        Iterable[ChargedGroup]
    ] = None,
    config: Optional[SaltBridgeConfig] = None,
    include_invalid: bool = False,
    include_pose_details: bool = True,
    include_model_details: bool = True,
    include_hotspot_details: bool = True,
) -> Dict[str, Any]:
    """
    Calculate complete salt-bridge statistics.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    cationic_groups
        Optional recognized cationic groups.
    anionic_groups
        Optional recognized anionic groups.
    config
        Salt-bridge configuration.
    include_invalid
        Whether invalid interactions should be included in descriptive
        distributions.
    include_pose_details
        Whether pose-level statistics should be calculated.
    include_model_details
        Whether model-level statistics should be calculated.
    include_hotspot_details
        Whether hotspot statistics should be calculated.

    Returns
    -------
    Dict[str, Any]
        Complete statistics dictionary.
    """

    resolved_config = resolve_config(config)

    interaction_list = list(
        interactions
    )

    normalized_interactions = (
        normalize_interaction_collection(
            interaction_list,
            include_invalid=include_invalid,
        )
    )

    statistics_data: Dict[str, Any] = {
        "counts": (
            calculate_interaction_count_statistics(
                interaction_list
            )
        ),
        "distances": (
            calculate_distance_statistics(
                normalized_interactions,
                include_invalid=True,
            )
        ),
        "scores": (
            calculate_score_statistics(
                normalized_interactions,
                include_invalid=True,
                include_zero=True,
            )
        ),
        "strength_distribution": (
            calculate_strength_distribution(
                interaction_list,
                include_rejected=True,
            )
        ),
        "group_type_distribution": (
            calculate_group_type_distribution(
                normalized_interactions,
                directional=True,
                include_invalid=True,
            )
        ),
        "recognition_source_distribution": (
            calculate_group_source_distribution(
                normalized_interactions,
                include_invalid=True,
            )
        ),
        "residues": (
            calculate_residue_participation_statistics(
                normalized_interactions,
                include_invalid=True,
            )
        ),
        "interfaces": (
            calculate_interface_statistics(
                normalized_interactions,
                include_invalid=True,
            )
        ),
    }

    if (
        cationic_groups is not None
        or anionic_groups is not None
    ):
        statistics_data[
            "recognized_groups"
        ] = calculate_recognized_group_statistics(
            cationic_groups or [],
            anionic_groups or [],
        )

    if include_pose_details:
        statistics_data[
            "poses"
        ] = calculate_pose_statistics(
            normalized_interactions,
            include_invalid=True,
        )

    if include_model_details:
        statistics_data[
            "models"
        ] = calculate_model_statistics(
            normalized_interactions,
            include_invalid=True,
        )

    if include_hotspot_details:
        statistics_data[
            "hotspots"
        ] = calculate_hotspot_statistics(
            normalized_interactions,
            resolved_config,
        )

    statistics_data["metadata"] = {
        "statistics_version": "1.0",
        "include_invalid": include_invalid,
        "input_interaction_count": len(
            interaction_list
        ),
        "analyzed_interaction_count": len(
            normalized_interactions
        ),
        "pose_details_included": (
            include_pose_details
        ),
        "model_details_included": (
            include_model_details
        ),
        "hotspot_details_included": (
            include_hotspot_details
        ),
    }

    return statistics_data


# =============================================================================
# 13.14. RESULT-LEVEL STATISTICS
# =============================================================================


def calculate_salt_bridge_result_statistics(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    in_place: bool = True,
    include_invalid: bool = False,
    include_pose_details: bool = True,
    include_model_details: bool = True,
    include_hotspot_details: bool = True,
) -> SaltBridgeResult:
    """
    Calculate and attach statistics to a SaltBridgeResult.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.
    in_place
        Whether the original result should be modified.
    include_invalid
        Whether invalid interactions should be included.
    include_pose_details
        Whether pose-level details should be stored.
    include_model_details
        Whether model-level details should be stored.
    include_hotspot_details
        Whether hotspot details should be stored.

    Returns
    -------
    SaltBridgeResult
        Result with populated statistics.
    """

    resolved_config = resolve_config(config)

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    target_result = result

    if not in_place:
        target_result = SaltBridgeResult(
            interactions=list(
                result.interactions
            ),
            cationic_groups=list(
                result.cationic_groups
            ),
            anionic_groups=list(
                result.anionic_groups
            ),
            statistics=dict(
                result.statistics
            ),
            warnings=list(
                result.warnings
            ),
            pose_id=result.pose_id,
            model_id=result.model_id,
            metadata=dict(
                result.metadata
            ),
        )

    statistics_data = (
        calculate_salt_bridge_statistics(
            target_result.interactions,
            cationic_groups=(
                target_result.cationic_groups
            ),
            anionic_groups=(
                target_result.anionic_groups
            ),
            config=resolved_config,
            include_invalid=include_invalid,
            include_pose_details=(
                include_pose_details
            ),
            include_model_details=(
                include_model_details
            ),
            include_hotspot_details=(
                include_hotspot_details
            ),
        )
    )

    target_result.statistics = (
        statistics_data
    )

    target_result.metadata[
        "statistics_completed"
    ] = True

    target_result.metadata[
        "statistics_interaction_count"
    ] = statistics_data[
        "metadata"
    ][
        "analyzed_interaction_count"
    ]

    target_result.metadata[
        "statistics_include_invalid"
    ] = include_invalid

    return target_result


# =============================================================================
# 13.15. COMPACT SUMMARY GENERATION
# =============================================================================


def build_compact_salt_bridge_summary(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    Build a compact machine-readable salt-bridge summary.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.

    Returns
    -------
    Dict[str, Any]
        Compact summary suitable for reports and DockModel integration.
    """

    if not isinstance(result, SaltBridgeResult):
        raise SaltBridgeDetectionError(
            "result must be a SaltBridgeResult instance."
        )

    if not result.statistics:
        calculate_salt_bridge_result_statistics(
            result,
            config,
            in_place=True,
        )

    counts = result.statistics.get(
        "counts",
        {},
    )

    distances = result.statistics.get(
        "distances",
        {},
    )

    scores = result.statistics.get(
        "scores",
        {},
    )

    residues = result.statistics.get(
        "residues",
        {},
    )

    interfaces = result.statistics.get(
        "interfaces",
        {},
    )

    hotspots = result.statistics.get(
        "hotspots",
        {},
    )

    minimum_distance_statistics = (
        distances.get(
            "minimum_atom_distance",
            {},
        )
    )

    top_hotspot = hotspots.get(
        "top_hotspot"
    )

    return {
        "interaction_count": counts.get(
            "valid_interaction_count",
            0,
        ),
        "rejected_candidate_count": counts.get(
            "rejected_interaction_count",
            0,
        ),
        "atomic_contact_count": counts.get(
            "total_atomic_contact_count",
            0,
        ),
        "total_score": scores.get(
            "sum",
            0.0,
        ),
        "mean_score": scores.get(
            "mean"
        ),
        "best_score": scores.get(
            "best_interaction_score"
        ),
        "best_interaction_id": scores.get(
            "best_interaction_id"
        ),
        "minimum_distance": (
            minimum_distance_statistics.get(
                "minimum"
            )
        ),
        "mean_distance": (
            minimum_distance_statistics.get(
                "mean"
            )
        ),
        "maximum_distance": (
            minimum_distance_statistics.get(
                "maximum"
            )
        ),
        "strong_count": (
            result.statistics
            .get(
                "strength_distribution",
                {},
            )
            .get(
                STRENGTH_STRONG,
                {},
            )
            .get(
                "count",
                0,
            )
        ),
        "moderate_count": (
            result.statistics
            .get(
                "strength_distribution",
                {},
            )
            .get(
                STRENGTH_MODERATE,
                {},
            )
            .get(
                "count",
                0,
            )
        ),
        "weak_count": (
            result.statistics
            .get(
                "strength_distribution",
                {},
            )
            .get(
                STRENGTH_WEAK,
                {},
            )
            .get(
                "count",
                0,
            )
        ),
        "residue_count": residues.get(
            "unique_residue_count",
            0,
        ),
        "cationic_residue_count": residues.get(
            "unique_cationic_residue_count",
            0,
        ),
        "anionic_residue_count": residues.get(
            "unique_anionic_residue_count",
            0,
        ),
        "intrachain_count": interfaces.get(
            "intrachain_interaction_count",
            0,
        ),
        "interchain_count": interfaces.get(
            "interchain_interaction_count",
            0,
        ),
        "hotspot_count": hotspots.get(
            "hotspot_count",
            0,
        ),
        "top_hotspot_label": (
            top_hotspot.get(
                "residue_label"
            )
            if top_hotspot is not None
            else None
        ),
        "top_hotspot_score": (
            top_hotspot.get(
                "hotspot_score"
            )
            if top_hotspot is not None
            else None
        ),
        "pose_id": result.pose_id,
        "model_id": result.model_id,
    }


def build_salt_bridge_text_summary(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
) -> str:
    """
    Build a concise human-readable salt-bridge summary.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.

    Returns
    -------
    str
        Human-readable summary.
    """

    summary = build_compact_salt_bridge_summary(
        result,
        config,
    )

    interaction_count = summary[
        "interaction_count"
    ]

    if interaction_count == 0:
        return (
            "No valid salt bridges were detected."
        )

    summary_parts = [
        (
            f"{interaction_count} valid salt bridge"
            f"{'' if interaction_count == 1 else 's'}"
        ),
        (
            f"{summary['strong_count']} strong"
        ),
        (
            f"{summary['moderate_count']} moderate"
        ),
        (
            f"{summary['weak_count']} weak"
        ),
        (
            f"total score {summary['total_score']:.3f}"
        ),
    ]

    if summary["minimum_distance"] is not None:
        summary_parts.append(
            "minimum distance "
            f"{summary['minimum_distance']:.3f} Å"
        )

    if summary["residue_count"] > 0:
        summary_parts.append(
            f"{summary['residue_count']} participating residues"
        )

    if summary["hotspot_count"] > 0:
        hotspot_label = (
            summary["top_hotspot_label"]
            or "unknown residue"
        )

        summary_parts.append(
            f"{summary['hotspot_count']} hotspots, "
            f"top hotspot {hotspot_label}"
        )

    return "; ".join(summary_parts) + "."


# =============================================================================
# 13.16. TABULAR SUMMARY RECORDS
# =============================================================================


def build_interaction_summary_record(
    interaction: SaltBridgeInteraction,
) -> Dict[str, Any]:
    """
    Build a flat summary record for one interaction.

    The record is suitable for table creation and later serialization.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Dict[str, Any]
        Flat interaction record.
    """

    return {
        "interaction_id": (
            interaction.interaction_id
        ),
        "interaction_type": (
            interaction.interaction_type
        ),
        "strength": interaction.strength,
        "score": interaction.score,
        "valid": interaction.geometry.valid,
        "rejection_reason": (
            interaction.geometry.rejection_reason
        ),
        "minimum_atom_distance": (
            interaction.geometry.minimum_atom_distance
        ),
        "center_distance": (
            interaction.geometry.center_distance
        ),
        "mean_atom_distance": (
            interaction.geometry.mean_atom_distance
        ),
        "maximum_atom_distance": (
            interaction.geometry.maximum_atom_distance
        ),
        "atomic_contact_count": (
            interaction.geometry.contact_count
        ),
        "cation_group_type": (
            interaction.cation.group_type
        ),
        "anion_group_type": (
            interaction.anion.group_type
        ),
        "cation_source": (
            interaction.cation.source
        ),
        "anion_source": (
            interaction.anion.source
        ),
        "cation_confidence": (
            interaction.cation.confidence
        ),
        "anion_confidence": (
            interaction.anion.confidence
        ),
        "cation_residue": make_residue_label(
            interaction.cation.residue,
            fallback="unknown_residue",
        ),
        "anion_residue": make_residue_label(
            interaction.anion.residue,
            fallback="unknown_residue",
        ),
        "cation_chain": get_chain_id(
            interaction.cation.residue
        ),
        "anion_chain": get_chain_id(
            interaction.anion.residue
        ),
        "pose_id": interaction.pose_id,
        "model_id": interaction.model_id,
    }


def build_interaction_summary_table(
    interactions: Iterable[SaltBridgeInteraction],
    *,
    include_invalid: bool = False,
    sort_by_score: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build flat summary records for multiple interactions.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.
    sort_by_score
        Whether interactions should be sorted by decreasing score.

    Returns
    -------
    List[Dict[str, Any]]
        Interaction summary records.
    """

    normalized_interactions = (
        normalize_interaction_collection(
            interactions,
            include_invalid=include_invalid,
        )
    )

    if sort_by_score:
        normalized_interactions = (
            sort_salt_bridges_by_score(
                normalized_interactions,
                descending=True,
            )
        )

    return [
        build_interaction_summary_record(
            interaction
        )
        for interaction in normalized_interactions
    ]


# =============================================================================
# 13.17. COMPLETE PIPELINE THROUGH STATISTICS
# =============================================================================


def analyze_salt_bridges_with_statistics(
    source: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
    deduplication_mode: Optional[str] = None,
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
) -> SaltBridgeResult:
    """
    Execute salt-bridge analysis through the statistics stage.

    The workflow includes:

    1. charged-group recognition;
    2. central detection;
    3. classification;
    4. scoring;
    5. deduplication;
    6. grouping;
    7. hotspot identification;
    8. statistical analysis;
    9. compact summary generation.

    Parameters
    ----------
    source
        Molecular source.
    config
        Salt-bridge configuration.
    pose_id
        Optional docking-pose identifier.
    model_id
        Optional molecular-model identifier.
    warnings
        Optional warning collector.
    deduplication_mode
        Optional interaction deduplication mode.
    store_full_groups
        Whether complete grouping mappings should be stored.
    include_invalid_statistics
        Whether invalid candidates should enter descriptive statistics.

    Returns
    -------
    SaltBridgeResult
        Fully analyzed result through Section 13.
    """

    resolved_config = resolve_config(config)

    result = analyze_grouped_salt_bridges(
        source,
        resolved_config,
        pose_id=pose_id,
        model_id=model_id,
        warnings=warnings,
        deduplication_mode=deduplication_mode,
        store_full_groups=store_full_groups,
    )

    result = calculate_salt_bridge_result_statistics(
        result,
        resolved_config,
        in_place=True,
        include_invalid=(
            include_invalid_statistics
        ),
        include_pose_details=True,
        include_model_details=True,
        include_hotspot_details=True,
    )

    result.metadata[
        "compact_summary"
    ] = build_compact_salt_bridge_summary(
        result,
        resolved_config,
    )

    result.metadata[
        "text_summary"
    ] = build_salt_bridge_text_summary(
        result,
        resolved_config,
    )

    return result


# =============================================================================
# 14. DOCKMODEL INTEGRATION
# =============================================================================


# =============================================================================
# 14.1. DOCKMODEL ACCESS UTILITIES
# =============================================================================


def is_dock_model_instance(
    value: Any,
) -> bool:
    """
    Return whether a value is a DockModel instance when DockModel is available.

    Duck-typed objects are not accepted by this function. Use
    ``is_dock_model_like`` when compatibility with custom containers is
    required.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    bool
        Whether the object is an instance of DockModel.
    """

    if DockModel is None:
        return False

    try:
        return isinstance(value, DockModel)

    except TypeError:
        return False


def is_dock_model_like(
    value: Any,
) -> bool:
    """
    Return whether an object can be used by the DockModel integration layer.

    A compatible object must be mutable through attributes or mapping keys and
    provide, directly or indirectly, a molecular source.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    bool
        Whether the object appears compatible.
    """

    if value is None:
        return False

    if is_dock_model_instance(value):
        return True

    if isinstance(value, Mapping):
        return True

    candidate_attributes = (
        "structure",
        "model",
        "molecule",
        "pose",
        "atoms",
        "residues",
        "saltbridge",
    )

    return any(
        hasattr(value, attribute_name)
        for attribute_name in candidate_attributes
    )


def get_dock_model_value(
    dock_model: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Read a DockModel attribute or mapping value safely.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    name
        Attribute or mapping key.
    default
        Value returned when the field is unavailable.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    if dock_model is None:
        return default

    if isinstance(dock_model, Mapping):
        return dock_model.get(
            name,
            default,
        )

    try:
        return getattr(
            dock_model,
            name,
            default,
        )

    except Exception:
        return default


def set_dock_model_value(
    dock_model: Any,
    name: str,
    value: Any,
    *,
    required: bool = True,
) -> bool:
    """
    Set a DockModel attribute or mutable mapping value.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    name
        Attribute or mapping key.
    value
        Value to assign.
    required
        Whether failure should raise an integration error.

    Returns
    -------
    bool
        Whether the value was assigned successfully.

    Raises
    ------
    DockModelSaltBridgeError
        If assignment fails and ``required`` is true.
    """

    if dock_model is None:
        if required:
            raise DockModelSaltBridgeError(
                "Cannot assign salt-bridge data to a null DockModel."
            )

        return False

    if isinstance(dock_model, dict):
        dock_model[name] = value
        return True

    try:
        setattr(
            dock_model,
            name,
            value,
        )

        return True

    except Exception as error:
        if required:
            raise DockModelSaltBridgeError(
                f"Could not assign DockModel field {name!r}."
            ) from error

        return False


def update_dock_model_mapping(
    dock_model: Any,
    name: str,
    values: Mapping[str, Any],
    *,
    preserve_existing: bool = True,
    required: bool = False,
) -> bool:
    """
    Update a mapping-like DockModel field.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    name
        Mapping attribute or key.
    values
        Values to merge.
    preserve_existing
        Whether existing keys should be preserved when conflicts occur.
    required
        Whether assignment failure should raise an error.

    Returns
    -------
    bool
        Whether the mapping was updated.
    """

    existing_value = get_dock_model_value(
        dock_model,
        name,
        None,
    )

    if isinstance(existing_value, Mapping):
        updated_mapping = dict(
            existing_value
        )

    else:
        updated_mapping = {}

    if preserve_existing:
        for key, value in values.items():
            updated_mapping.setdefault(
                key,
                value,
            )

    else:
        updated_mapping.update(
            values
        )

    return set_dock_model_value(
        dock_model,
        name,
        updated_mapping,
        required=required,
    )


# =============================================================================
# 14.2. MOLECULAR SOURCE RESOLUTION
# =============================================================================


DEFAULT_DOCK_MODEL_SOURCE_FIELDS: Tuple[str, ...] = (
    "structure",
    "molecular_structure",
    "chimera_model",
    "atomic_model",
    "model",
    "molecule",
    "pose",
    "complex",
    "receptor_ligand_complex",
    "atoms",
    "residues",
)


def resolve_dock_model_source(
    dock_model: Any,
    *,
    source: Any = None,
    source_fields: Optional[Iterable[str]] = None,
) -> Any:
    """
    Resolve the molecular source associated with a DockModel.

    An explicitly supplied source has priority. Otherwise, common DockModel
    fields are inspected in sequence.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    source
        Optional explicit molecular source.
    source_fields
        Optional ordered field names to inspect.

    Returns
    -------
    Any
        Resolved molecular source.

    Raises
    ------
    DockModelSaltBridgeError
        If no molecular source can be resolved.
    """

    if source is not None:
        return source

    if dock_model is None:
        raise DockModelSaltBridgeError(
            "A DockModel is required to resolve a molecular source."
        )

    candidate_fields = tuple(
        source_fields
        or DEFAULT_DOCK_MODEL_SOURCE_FIELDS
    )

    for field_name in candidate_fields:
        candidate_source = get_dock_model_value(
            dock_model,
            field_name,
            None,
        )

        if candidate_source is not None:
            return candidate_source

    if (
        hasattr(dock_model, "atoms")
        or hasattr(dock_model, "residues")
    ):
        return dock_model

    if isinstance(dock_model, Mapping):
        if (
            "atoms" in dock_model
            or "residues" in dock_model
        ):
            return dock_model

    raise DockModelSaltBridgeError(
        "No molecular source could be resolved from the DockModel."
    )


def resolve_dock_model_pose_id(
    dock_model: Any,
    *,
    pose_id: Optional[Union[str, int]] = None,
) -> Optional[Union[str, int]]:
    """
    Resolve a pose identifier from a DockModel.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    pose_id
        Optional explicit pose identifier.

    Returns
    -------
    Optional[Union[str, int]]
        Resolved pose identifier.
    """

    if pose_id is not None:
        return normalize_pose_identifier(
            pose_id
        )

    candidate_fields = (
        "pose_id",
        "pose_number",
        "pose_index",
        "rank",
        "mode",
        "conformation_id",
    )

    for field_name in candidate_fields:
        value = get_dock_model_value(
            dock_model,
            field_name,
            None,
        )

        if value is not None:
            return normalize_pose_identifier(
                value
            )

    return None


def resolve_dock_model_model_id(
    dock_model: Any,
    *,
    model_id: Optional[Union[str, int]] = None,
) -> Optional[Union[str, int]]:
    """
    Resolve a model identifier from a DockModel.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    model_id
        Optional explicit model identifier.

    Returns
    -------
    Optional[Union[str, int]]
        Resolved model identifier.
    """

    if model_id is not None:
        return normalize_model_identifier(
            model_id
        )

    candidate_fields = (
        "model_id",
        "identifier",
        "id",
        "name",
        "model_name",
        "title",
    )

    for field_name in candidate_fields:
        value = get_dock_model_value(
            dock_model,
            field_name,
            None,
        )

        if value is not None:
            return normalize_model_identifier(
                value
            )

    return None


# =============================================================================
# 14.3. RESULT ATTACHMENT
# =============================================================================


def merge_salt_bridge_interactions(
    existing: Iterable[SaltBridgeInteraction],
    new: Iterable[SaltBridgeInteraction],
    config: Optional[SaltBridgeConfig] = None,
) -> List[SaltBridgeInteraction]:
    """
    Merge existing and newly detected interactions.

    The combined list is deduplicated while preserving pose and model
    boundaries.

    Parameters
    ----------
    existing
        Existing interactions.
    new
        Newly detected interactions.
    config
        Salt-bridge configuration.

    Returns
    -------
    List[SaltBridgeInteraction]
        Merged and deduplicated interactions.
    """

    combined_interactions = [
        interaction
        for interaction in (
            list(existing)
            + list(new)
        )
        if isinstance(
            interaction,
            SaltBridgeInteraction,
        )
    ]

    return deduplicate_salt_bridge_interactions(
        combined_interactions,
        config,
        include_pose=True,
        include_model=True,
        merge_metadata=True,
    )


def attach_salt_bridge_results(
    dock_model: Any,
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    attribute_name: str = "saltbridge",
    preserve_existing: bool = True,
    attach_result_object: bool = True,
    attach_statistics: bool = True,
    attach_summary: bool = True,
) -> Any:
    """
    Attach salt-bridge analysis results to a DockModel.

    The primary ``saltbridge`` field receives a list of
    ``SaltBridgeInteraction`` objects.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    result
        Salt-bridge analysis result.
    config
        Salt-bridge configuration.
    attribute_name
        DockModel field receiving the interaction list.
    preserve_existing
        Whether existing interactions should be merged instead of replaced.
    attach_result_object
        Whether the complete SaltBridgeResult should be stored.
    attach_statistics
        Whether statistics should be stored separately.
    attach_summary
        Whether compact and textual summaries should be stored.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    resolved_config = resolve_config(config)

    if not is_dock_model_like(dock_model):
        raise DockModelSaltBridgeError(
            "dock_model is not compatible with DockModel integration."
        )

    if not isinstance(result, SaltBridgeResult):
        raise DockModelSaltBridgeError(
            "result must be a SaltBridgeResult instance."
        )

    existing_interactions = get_dock_model_value(
        dock_model,
        attribute_name,
        [],
    )

    if not isinstance(
        existing_interactions,
        (list, tuple),
    ):
        existing_interactions = []

    if preserve_existing:
        interactions_to_attach = (
            merge_salt_bridge_interactions(
                existing_interactions,
                result.interactions,
                resolved_config,
            )
        )

    else:
        interactions_to_attach = list(
            result.interactions
        )

    set_dock_model_value(
        dock_model,
        attribute_name,
        interactions_to_attach,
        required=True,
    )

    if attach_result_object:
        set_dock_model_value(
            dock_model,
            "saltbridge_result",
            result,
            required=False,
        )

    if attach_statistics:
        set_dock_model_value(
            dock_model,
            "saltbridge_statistics",
            dict(result.statistics),
            required=False,
        )

    if attach_summary:
        compact_summary = (
            result.metadata.get(
                "compact_summary"
            )
            or build_compact_salt_bridge_summary(
                result,
                resolved_config,
            )
        )

        text_summary = (
            result.metadata.get(
                "text_summary"
            )
            or build_salt_bridge_text_summary(
                result,
                resolved_config,
            )
        )

        set_dock_model_value(
            dock_model,
            "saltbridge_summary",
            compact_summary,
            required=False,
        )

        set_dock_model_value(
            dock_model,
            "saltbridge_text_summary",
            text_summary,
            required=False,
        )

    set_dock_model_value(
        dock_model,
        "saltbridge_analyzed",
        True,
        required=False,
    )

    return dock_model


# =============================================================================
# 14.4. DOCKMODEL STATISTICS UPDATE
# =============================================================================


def build_dock_model_salt_bridge_statistics(
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    Build DockModel-compatible salt-bridge statistics.

    Parameters
    ----------
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.

    Returns
    -------
    Dict[str, Any]
        Compact DockModel statistics.
    """

    if not result.statistics:
        calculate_salt_bridge_result_statistics(
            result,
            config,
            in_place=True,
        )

    summary = build_compact_salt_bridge_summary(
        result,
        config,
    )

    return {
        "count": summary[
            "interaction_count"
        ],
        "rejected_count": summary[
            "rejected_candidate_count"
        ],
        "atomic_contact_count": summary[
            "atomic_contact_count"
        ],
        "score": summary[
            "total_score"
        ],
        "mean_score": summary[
            "mean_score"
        ],
        "best_score": summary[
            "best_score"
        ],
        "minimum_distance": summary[
            "minimum_distance"
        ],
        "mean_distance": summary[
            "mean_distance"
        ],
        "maximum_distance": summary[
            "maximum_distance"
        ],
        "strong_count": summary[
            "strong_count"
        ],
        "moderate_count": summary[
            "moderate_count"
        ],
        "weak_count": summary[
            "weak_count"
        ],
        "residue_count": summary[
            "residue_count"
        ],
        "hotspot_count": summary[
            "hotspot_count"
        ],
        "intrachain_count": summary[
            "intrachain_count"
        ],
        "interchain_count": summary[
            "interchain_count"
        ],
    }


def update_dock_model_statistics(
    dock_model: Any,
    result: SaltBridgeResult,
    config: Optional[SaltBridgeConfig] = None,
    *,
    statistics_attribute: str = "statistics",
    preserve_existing: bool = True,
) -> Any:
    """
    Update the general DockModel statistics mapping.

    Salt-bridge statistics are stored under the ``"saltbridge"`` key so
    unrelated analysis statistics remain intact.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    result
        Salt-bridge result.
    config
        Salt-bridge configuration.
    statistics_attribute
        General statistics mapping field.
    preserve_existing
        Whether unrelated existing statistics should be preserved.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    salt_bridge_statistics = (
        build_dock_model_salt_bridge_statistics(
            result,
            config,
        )
    )

    existing_statistics = get_dock_model_value(
        dock_model,
        statistics_attribute,
        {},
    )

    if isinstance(existing_statistics, Mapping):
        updated_statistics = dict(
            existing_statistics
        )

    else:
        updated_statistics = {}

    if (
        preserve_existing
        and "saltbridge" in updated_statistics
        and isinstance(
            updated_statistics["saltbridge"],
            Mapping,
        )
    ):
        merged_salt_bridge_statistics = dict(
            updated_statistics[
                "saltbridge"
            ]
        )

        merged_salt_bridge_statistics.update(
            salt_bridge_statistics
        )

        updated_statistics[
            "saltbridge"
        ] = merged_salt_bridge_statistics

    else:
        updated_statistics[
            "saltbridge"
        ] = salt_bridge_statistics

    set_dock_model_value(
        dock_model,
        statistics_attribute,
        updated_statistics,
        required=False,
    )

    return dock_model


# =============================================================================
# 14.5. DOCKMODEL SCORE UPDATE
# =============================================================================


def get_result_salt_bridge_score(
    result: SaltBridgeResult,
) -> float:
    """
    Return the total valid salt-bridge score from a result.

    Parameters
    ----------
    result
        Salt-bridge result.

    Returns
    -------
    float
        Total non-negative salt-bridge score.
    """

    if not isinstance(result, SaltBridgeResult):
        raise DockModelSaltBridgeError(
            "result must be a SaltBridgeResult instance."
        )

    return sum(
        max(
            0.0,
            safe_float(
                interaction.score,
                default=0.0,
            ) or 0.0,
        )
        for interaction in result.interactions
        if interaction.geometry.valid
    )


def update_dock_model_salt_bridge_score(
    dock_model: Any,
    result: SaltBridgeResult,
    *,
    dedicated_attribute: str = "saltbridge_score",
    update_total_score: bool = False,
    total_score_attribute: str = "score",
    total_score_mode: str = "add",
) -> Any:
    """
    Update DockModel salt-bridge and optional total scores.

    The dedicated salt-bridge score is always stored when possible. Updating
    the global DockModel score is optional because different docking pipelines
    may use incompatible score conventions.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    result
        Salt-bridge result.
    dedicated_attribute
        Attribute receiving the salt-bridge score.
    update_total_score
        Whether the general DockModel score should be modified.
    total_score_attribute
        General score attribute.
    total_score_mode
        General score update mode. Supported values are ``"add"``,
        ``"replace"``, and ``"subtract"``.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    salt_bridge_score = (
        get_result_salt_bridge_score(
            result
        )
    )

    set_dock_model_value(
        dock_model,
        dedicated_attribute,
        salt_bridge_score,
        required=False,
    )

    if not update_total_score:
        return dock_model

    normalized_mode = normalize_text(
        total_score_mode,
        default="add",
        lowercase=True,
    )

    current_total_score = safe_float(
        get_dock_model_value(
            dock_model,
            total_score_attribute,
            0.0,
        ),
        default=0.0,
    )

    current_total_score = (
        current_total_score or 0.0
    )

    if normalized_mode == "add":
        updated_total_score = (
            current_total_score
            + salt_bridge_score
        )

    elif normalized_mode == "subtract":
        updated_total_score = (
            current_total_score
            - salt_bridge_score
        )

    elif normalized_mode == "replace":
        updated_total_score = (
            salt_bridge_score
        )

    else:
        raise DockModelSaltBridgeError(
            f"Unsupported total score mode: {total_score_mode!r}."
        )

    set_dock_model_value(
        dock_model,
        total_score_attribute,
        updated_total_score,
        required=False,
    )

    return dock_model


# =============================================================================
# 14.6. SINGLE DOCKMODEL ANALYSIS
# =============================================================================


def analyze_dock_model_salt_bridges(
    dock_model: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    source: Any = None,
    source_fields: Optional[Iterable[str]] = None,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    warnings: Optional[List[str]] = None,
    preserve_existing: bool = True,
    attach_result_object: bool = True,
    update_statistics: bool = True,
    update_score: bool = True,
    update_total_score: bool = False,
    total_score_mode: str = "add",
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
) -> SaltBridgeResult:
    """
    Analyze salt bridges for one DockModel and attach the result.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    config
        Salt-bridge configuration.
    source
        Optional explicit molecular source.
    source_fields
        Optional ordered DockModel source fields.
    pose_id
        Optional explicit pose identifier.
    model_id
        Optional explicit model identifier.
    warnings
        Optional warning collector.
    preserve_existing
        Whether existing salt-bridge interactions should be preserved.
    attach_result_object
        Whether the complete result should be stored.
    update_statistics
        Whether DockModel statistics should be updated.
    update_score
        Whether the dedicated salt-bridge score should be updated.
    update_total_score
        Whether the general DockModel score should be modified.
    total_score_mode
        General score update mode.
    store_full_groups
        Whether full grouping mappings should be retained.
    include_invalid_statistics
        Whether invalid candidates should enter descriptive statistics.

    Returns
    -------
    SaltBridgeResult
        Salt-bridge result attached to the DockModel.
    """

    resolved_config = resolve_config(config)

    if not is_dock_model_like(dock_model):
        raise DockModelSaltBridgeError(
            "The supplied object is not DockModel-compatible."
        )

    resolved_source = resolve_dock_model_source(
        dock_model,
        source=source,
        source_fields=source_fields,
    )

    resolved_pose_id = (
        resolve_dock_model_pose_id(
            dock_model,
            pose_id=pose_id,
        )
    )

    resolved_model_id = (
        resolve_dock_model_model_id(
            dock_model,
            model_id=model_id,
        )
    )

    try:
        result = analyze_salt_bridges_with_statistics(
            resolved_source,
            resolved_config,
            pose_id=resolved_pose_id,
            model_id=resolved_model_id,
            warnings=warnings,
            store_full_groups=store_full_groups,
            include_invalid_statistics=(
                include_invalid_statistics
            ),
        )

        attach_salt_bridge_results(
            dock_model,
            result,
            resolved_config,
            preserve_existing=preserve_existing,
            attach_result_object=(
                attach_result_object
            ),
            attach_statistics=True,
            attach_summary=True,
        )

        if update_statistics:
            update_dock_model_statistics(
                dock_model,
                result,
                resolved_config,
                preserve_existing=True,
            )

        if update_score:
            update_dock_model_salt_bridge_score(
                dock_model,
                result,
                update_total_score=(
                    update_total_score
                ),
                total_score_mode=(
                    total_score_mode
                ),
            )

        return result

    except SaltBridgeError:
        raise

    except Exception as error:
        raise DockModelSaltBridgeError(
            "Unexpected failure during DockModel salt-bridge analysis."
        ) from error


# =============================================================================
# 14.7. MULTIPLE DOCKMODEL ANALYSIS
# =============================================================================


def analyze_multiple_dock_models_salt_bridges(
    dock_models: Iterable[Any],
    config: Optional[SaltBridgeConfig] = None,
    *,
    preserve_existing: bool = True,
    attach_result_object: bool = True,
    update_statistics: bool = True,
    update_score: bool = True,
    update_total_score: bool = False,
    total_score_mode: str = "add",
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
    continue_on_error: bool = True,
    warnings: Optional[List[str]] = None,
) -> List[SaltBridgeResult]:
    """
    Analyze salt bridges for multiple DockModel objects.

    Parameters
    ----------
    dock_models
        DockModel-like objects.
    config
        Salt-bridge configuration.
    preserve_existing
        Whether existing interactions should be preserved.
    attach_result_object
        Whether complete results should be attached.
    update_statistics
        Whether general DockModel statistics should be updated.
    update_score
        Whether dedicated salt-bridge scores should be updated.
    update_total_score
        Whether general DockModel scores should be modified.
    total_score_mode
        General score update mode.
    store_full_groups
        Whether complete grouping mappings should be retained.
    include_invalid_statistics
        Whether invalid candidates should enter descriptive statistics.
    continue_on_error
        Whether processing should continue after a model-level failure.
    warnings
        Optional warning collector.

    Returns
    -------
    List[SaltBridgeResult]
        Successfully generated results.
    """

    resolved_config = resolve_config(config)
    result_list: List[SaltBridgeResult] = []

    for model_index, dock_model in enumerate(
        dock_models,
        start=1,
    ):
        try:
            result = analyze_dock_model_salt_bridges(
                dock_model,
                resolved_config,
                preserve_existing=preserve_existing,
                attach_result_object=(
                    attach_result_object
                ),
                update_statistics=(
                    update_statistics
                ),
                update_score=update_score,
                update_total_score=(
                    update_total_score
                ),
                total_score_mode=(
                    total_score_mode
                ),
                store_full_groups=(
                    store_full_groups
                ),
                include_invalid_statistics=(
                    include_invalid_statistics
                ),
                warnings=warnings,
            )

            result.metadata[
                "dock_model_batch_index"
            ] = model_index

            result_list.append(
                result
            )

        except SaltBridgeError as error:
            message = (
                "DockModel salt-bridge analysis failed "
                f"for model index {model_index}: {error}"
            )

            if warnings is not None:
                warnings.append(
                    message
                )

            if not continue_on_error:
                raise

    return result_list


# =============================================================================
# 14.8. DOCKMODEL RESULT ACCESS
# =============================================================================


def get_dock_model_salt_bridges(
    dock_model: Any,
    *,
    attribute_name: str = "saltbridge",
    valid_only: bool = False,
) -> List[SaltBridgeInteraction]:
    """
    Return salt-bridge interactions attached to a DockModel.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    attribute_name
        Interaction-list attribute.
    valid_only
        Whether invalid candidates should be removed.

    Returns
    -------
    List[SaltBridgeInteraction]
        Attached interactions.
    """

    attached_value = get_dock_model_value(
        dock_model,
        attribute_name,
        [],
    )

    if not isinstance(
        attached_value,
        (list, tuple),
    ):
        return []

    interactions = [
        interaction
        for interaction in attached_value
        if isinstance(
            interaction,
            SaltBridgeInteraction,
        )
    ]

    if valid_only:
        interactions = [
            interaction
            for interaction in interactions
            if interaction.geometry.valid
        ]

    return interactions


def get_dock_model_salt_bridge_result(
    dock_model: Any,
) -> Optional[SaltBridgeResult]:
    """
    Return the complete SaltBridgeResult attached to a DockModel.

    Parameters
    ----------
    dock_model
        DockModel-like object.

    Returns
    -------
    Optional[SaltBridgeResult]
        Attached result or ``None``.
    """

    result = get_dock_model_value(
        dock_model,
        "saltbridge_result",
        None,
    )

    if isinstance(
        result,
        SaltBridgeResult,
    ):
        return result

    return None


def clear_dock_model_salt_bridges(
    dock_model: Any,
    *,
    clear_score: bool = True,
    clear_statistics: bool = True,
) -> Any:
    """
    Remove salt-bridge data from a DockModel.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    clear_score
        Whether dedicated salt-bridge score should be reset.
    clear_statistics
        Whether dedicated salt-bridge statistics should be cleared.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    set_dock_model_value(
        dock_model,
        "saltbridge",
        [],
        required=True,
    )

    set_dock_model_value(
        dock_model,
        "saltbridge_result",
        None,
        required=False,
    )

    set_dock_model_value(
        dock_model,
        "saltbridge_summary",
        {},
        required=False,
    )

    set_dock_model_value(
        dock_model,
        "saltbridge_text_summary",
        "",
        required=False,
    )

    set_dock_model_value(
        dock_model,
        "saltbridge_analyzed",
        False,
        required=False,
    )

    if clear_score:
        set_dock_model_value(
            dock_model,
            "saltbridge_score",
            0.0,
            required=False,
        )

    if clear_statistics:
        set_dock_model_value(
            dock_model,
            "saltbridge_statistics",
            {},
            required=False,
        )

        existing_statistics = get_dock_model_value(
            dock_model,
            "statistics",
            None,
        )

        if isinstance(
            existing_statistics,
            Mapping,
        ):
            updated_statistics = dict(
                existing_statistics
            )

            updated_statistics.pop(
                "saltbridge",
                None,
            )

            set_dock_model_value(
                dock_model,
                "statistics",
                updated_statistics,
                required=False,
            )

    return dock_model


# =============================================================================
# 14.9. BATCH SUMMARY
# =============================================================================


def summarize_dock_model_salt_bridge_results(
    results: Iterable[SaltBridgeResult],
) -> Dict[str, Any]:
    """
    Summarize salt-bridge results from multiple DockModel objects.

    Parameters
    ----------
    results
        DockModel salt-bridge results.

    Returns
    -------
    Dict[str, Any]
        Batch-level summary.
    """

    result_list = [
        result
        for result in results
        if isinstance(
            result,
            SaltBridgeResult,
        )
    ]

    interaction_counts = [
        len(result.valid_interactions)
        for result in result_list
    ]

    total_scores = [
        get_result_salt_bridge_score(
            result
        )
        for result in result_list
    ]

    minimum_distances: List[float] = []

    for result in result_list:
        valid_distances = [
            interaction.distance
            for interaction in result.valid_interactions
            if safe_float(
                interaction.distance
            ) is not None
        ]

        if valid_distances:
            minimum_distances.append(
                min(valid_distances)
            )

    best_result = None

    if result_list:
        best_result = max(
            result_list,
            key=lambda result: (
                get_result_salt_bridge_score(
                    result
                ),
                len(result.valid_interactions),
                -min(
                    (
                        interaction.distance
                        for interaction
                        in result.valid_interactions
                    ),
                    default=math.inf,
                ),
            ),
        )

    return {
        "model_count": len(
            result_list
        ),
        "total_interaction_count": sum(
            interaction_counts
        ),
        "total_score": sum(
            total_scores
        ),
        "interactions_per_model": (
            calculate_numeric_statistics(
                interaction_counts
            )
        ),
        "scores_per_model": (
            calculate_numeric_statistics(
                total_scores
            )
        ),
        "minimum_distances_per_model": (
            calculate_numeric_statistics(
                minimum_distances
            )
        ),
        "best_model_id": (
            best_result.model_id
            if best_result is not None
            else None
        ),
        "best_pose_id": (
            best_result.pose_id
            if best_result is not None
            else None
        ),
        "best_model_score": (
            get_result_salt_bridge_score(
                best_result
            )
            if best_result is not None
            else None
        ),
    }



# =============================================================================
# 15. MULTIPOSE ANALYSIS
# =============================================================================


# =============================================================================
# 15.1. MULTIPOSE INPUT NORMALIZATION
# =============================================================================


def normalize_pose_collection(
    poses: Iterable[Any],
) -> List[Any]:
    """
    Normalize a collection of docking poses.

    Parameters
    ----------
    poses
        Pose-like molecular sources or DockModel-like objects.

    Returns
    -------
    List[Any]
        Materialized pose collection.

    Raises
    ------
    SaltBridgeDetectionError
        If the pose collection is invalid or empty.
    """

    if poses is None:
        raise SaltBridgeDetectionError(
            "A pose collection is required."
        )

    try:
        pose_list = list(poses)

    except TypeError as error:
        raise SaltBridgeDetectionError(
            "poses must be an iterable collection."
        ) from error

    if not pose_list:
        raise SaltBridgeDetectionError(
            "The pose collection cannot be empty."
        )

    return pose_list


def make_multipose_pose_id(
    pose: Any,
    pose_index: int,
    *,
    explicit_pose_id: Optional[Union[str, int]] = None,
) -> Union[str, int]:
    """
    Resolve a stable pose identifier.

    Parameters
    ----------
    pose
        Pose-like object.
    pose_index
        One-based pose position in the input collection.
    explicit_pose_id
        Optional explicit pose identifier.

    Returns
    -------
    Union[str, int]
        Resolved pose identifier.
    """

    if explicit_pose_id is not None:
        normalized_pose_id = normalize_pose_identifier(
            explicit_pose_id
        )

        if normalized_pose_id is not None:
            return normalized_pose_id

    if is_dock_model_like(pose):
        resolved_pose_id = resolve_dock_model_pose_id(
            pose
        )

        if resolved_pose_id is not None:
            return resolved_pose_id

    candidate_fields = (
        "pose_id",
        "pose_number",
        "pose_index",
        "rank",
        "mode",
        "state_id",
        "conformation_id",
    )

    for field_name in candidate_fields:
        candidate_value = get_value(
            pose,
            field_name,
            None,
        )

        if candidate_value is not None:
            normalized_pose_id = (
                normalize_pose_identifier(
                    candidate_value
                )
            )

            if normalized_pose_id is not None:
                return normalized_pose_id

    return pose_index


def make_multipose_model_id(
    pose: Any,
    pose_index: int,
    *,
    explicit_model_id: Optional[Union[str, int]] = None,
    default_prefix: str = "model",
) -> Union[str, int]:
    """
    Resolve a stable model identifier for one pose.

    Parameters
    ----------
    pose
        Pose-like object.
    pose_index
        One-based pose position.
    explicit_model_id
        Optional explicit model identifier.
    default_prefix
        Prefix used for generated identifiers.

    Returns
    -------
    Union[str, int]
        Resolved model identifier.
    """

    if explicit_model_id is not None:
        normalized_model_id = (
            normalize_model_identifier(
                explicit_model_id
            )
        )

        if normalized_model_id is not None:
            return normalized_model_id

    if is_dock_model_like(pose):
        resolved_model_id = resolve_dock_model_model_id(
            pose
        )

        if resolved_model_id is not None:
            return resolved_model_id

    candidate_fields = (
        "model_id",
        "identifier",
        "name",
        "title",
    )

    for field_name in candidate_fields:
        candidate_value = get_value(
            pose,
            field_name,
            None,
        )

        if candidate_value is not None:
            normalized_model_id = (
                normalize_model_identifier(
                    candidate_value
                )
            )

            if normalized_model_id is not None:
                return normalized_model_id

    return f"{default_prefix}_{pose_index:04d}"


def resolve_multipose_source(
    pose: Any,
) -> Any:
    """
    Resolve the molecular source for one pose.

    DockModel-like objects are resolved through the DockModel integration
    layer. Other objects are treated directly as molecular sources.

    Parameters
    ----------
    pose
        Pose-like source or DockModel-like object.

    Returns
    -------
    Any
        Molecular source.
    """

    if is_dock_model_like(pose):
        try:
            return resolve_dock_model_source(
                pose
            )

        except DockModelSaltBridgeError:
            pass

    return pose


def normalize_pose_id_mapping(
    pose_count: int,
    pose_ids: Optional[
        Union[
            Sequence[Optional[Union[str, int]]],
            Mapping[int, Optional[Union[str, int]]],
        ]
    ] = None,
) -> Dict[int, Optional[Union[str, int]]]:
    """
    Normalize optional pose identifiers into an index-based mapping.

    Indices are one-based to match user-facing pose numbering.

    Parameters
    ----------
    pose_count
        Number of poses.
    pose_ids
        Optional sequence or mapping of pose identifiers.

    Returns
    -------
    Dict[int, Optional[Union[str, int]]]
        One-based pose-index mapping.
    """

    if pose_ids is None:
        return {
            index: None
            for index in range(
                1,
                pose_count + 1,
            )
        }

    if isinstance(pose_ids, Mapping):
        return {
            index: pose_ids.get(
                index
            )
            for index in range(
                1,
                pose_count + 1,
            )
        }

    pose_id_list = list(
        pose_ids
    )

    if len(pose_id_list) != pose_count:
        raise SaltBridgeDetectionError(
            "pose_ids must contain one identifier per pose."
        )

    return {
        index: pose_id_list[
            index - 1
        ]
        for index in range(
            1,
            pose_count + 1,
        )
    }


# =============================================================================
# 15.2. SINGLE-POSE EXECUTION WITHIN MULTIPOSE ANALYSIS
# =============================================================================


def analyze_single_multipose_entry(
    pose: Any,
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_index: int,
    pose_id: Optional[Union[str, int]] = None,
    model_id: Optional[Union[str, int]] = None,
    attach_to_dock_model: bool = True,
    preserve_existing: bool = False,
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
    warnings: Optional[List[str]] = None,
) -> SaltBridgeResult:
    """
    Analyze one entry from a multipose collection.

    Parameters
    ----------
    pose
        Pose-like source or DockModel-like object.
    config
        Salt-bridge configuration.
    pose_index
        One-based pose position.
    pose_id
        Optional explicit pose identifier.
    model_id
        Optional explicit model identifier.
    attach_to_dock_model
        Whether results should be attached when the pose is a DockModel.
    preserve_existing
        Whether pre-existing DockModel salt bridges should be preserved.
    store_full_groups
        Whether complete grouping mappings should be stored.
    include_invalid_statistics
        Whether invalid candidates should enter descriptive statistics.
    warnings
        Optional warning collector.

    Returns
    -------
    SaltBridgeResult
        Analysis result for one pose.
    """

    resolved_config = resolve_config(
        config
    )

    resolved_pose_id = make_multipose_pose_id(
        pose,
        pose_index,
        explicit_pose_id=pose_id,
    )

    resolved_model_id = make_multipose_model_id(
        pose,
        pose_index,
        explicit_model_id=model_id,
    )

    if (
        attach_to_dock_model
        and is_dock_model_like(pose)
    ):
        result = analyze_dock_model_salt_bridges(
            pose,
            resolved_config,
            pose_id=resolved_pose_id,
            model_id=resolved_model_id,
            warnings=warnings,
            preserve_existing=preserve_existing,
            attach_result_object=True,
            update_statistics=True,
            update_score=True,
            update_total_score=False,
            store_full_groups=store_full_groups,
            include_invalid_statistics=(
                include_invalid_statistics
            ),
        )

    else:
        molecular_source = resolve_multipose_source(
            pose
        )

        result = analyze_salt_bridges_with_statistics(
            molecular_source,
            resolved_config,
            pose_id=resolved_pose_id,
            model_id=resolved_model_id,
            warnings=warnings,
            store_full_groups=store_full_groups,
            include_invalid_statistics=(
                include_invalid_statistics
            ),
        )

    result.metadata[
        "multipose_pose_index"
    ] = pose_index

    result.metadata[
        "multipose_analysis"
    ] = True

    return result


# =============================================================================
# 15.3. MULTIPOSE EXECUTION
# =============================================================================


def analyze_multiple_poses_salt_bridges(
    poses: Iterable[Any],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_ids: Optional[
        Union[
            Sequence[Optional[Union[str, int]]],
            Mapping[int, Optional[Union[str, int]]],
        ]
    ] = None,
    model_ids: Optional[
        Union[
            Sequence[Optional[Union[str, int]]],
            Mapping[int, Optional[Union[str, int]]],
        ]
    ] = None,
    attach_to_dock_models: bool = True,
    preserve_existing: bool = False,
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
    continue_on_error: bool = True,
    warnings: Optional[List[str]] = None,
) -> List[SaltBridgeResult]:
    """
    Analyze salt bridges across multiple docking poses.

    Each pose is analyzed independently. Interactions are not deduplicated
    across different poses.

    Parameters
    ----------
    poses
        Pose-like sources or DockModel-like objects.
    config
        Salt-bridge configuration.
    pose_ids
        Optional pose identifiers.
    model_ids
        Optional model identifiers.
    attach_to_dock_models
        Whether results should be attached to DockModel-like inputs.
    preserve_existing
        Whether existing DockModel interactions should be preserved.
    store_full_groups
        Whether complete group mappings should be stored.
    include_invalid_statistics
        Whether invalid candidates should enter descriptive statistics.
    continue_on_error
        Whether analysis should continue after one pose fails.
    warnings
        Optional warning collector.

    Returns
    -------
    List[SaltBridgeResult]
        Successfully analyzed pose results.
    """

    resolved_config = resolve_config(
        config
    )

    pose_list = normalize_pose_collection(
        poses
    )

    pose_id_mapping = normalize_pose_id_mapping(
        len(pose_list),
        pose_ids,
    )

    model_id_mapping = normalize_pose_id_mapping(
        len(pose_list),
        model_ids,
    )

    result_list: List[
        SaltBridgeResult
    ] = []

    for pose_index, pose in enumerate(
        pose_list,
        start=1,
    ):
        try:
            result = analyze_single_multipose_entry(
                pose,
                resolved_config,
                pose_index=pose_index,
                pose_id=pose_id_mapping[
                    pose_index
                ],
                model_id=model_id_mapping[
                    pose_index
                ],
                attach_to_dock_model=(
                    attach_to_dock_models
                ),
                preserve_existing=(
                    preserve_existing
                ),
                store_full_groups=(
                    store_full_groups
                ),
                include_invalid_statistics=(
                    include_invalid_statistics
                ),
                warnings=warnings,
            )

            result_list.append(
                result
            )

        except SaltBridgeError as error:
            message = (
                "Salt-bridge analysis failed for "
                f"pose index {pose_index}: {error}"
            )

            if warnings is not None:
                warnings.append(
                    message
                )

            if not continue_on_error:
                raise

        except Exception as error:
            wrapped_error = SaltBridgeDetectionError(
                "Unexpected multipose analysis failure "
                f"at pose index {pose_index}."
            )

            if warnings is not None:
                warnings.append(
                    f"{wrapped_error}: {error}"
                )

            if not continue_on_error:
                raise wrapped_error from error

    return result_list


# =============================================================================
# 15.4. MULTIPOSE INTERACTION COLLECTION
# =============================================================================


def collect_multipose_interactions(
    results: Iterable[SaltBridgeResult],
    *,
    valid_only: bool = True,
) -> List[SaltBridgeInteraction]:
    """
    Collect interactions from multiple pose results.

    No cross-pose deduplication is performed.

    Parameters
    ----------
    results
        Pose-level salt-bridge results.
    valid_only
        Whether invalid interactions should be excluded.

    Returns
    -------
    List[SaltBridgeInteraction]
        Combined interaction collection.
    """

    collected_interactions: List[
        SaltBridgeInteraction
    ] = []

    for result in results:
        if not isinstance(
            result,
            SaltBridgeResult,
        ):
            raise SaltBridgeDetectionError(
                "All results must be SaltBridgeResult instances."
            )

        for interaction in result.interactions:
            if (
                valid_only
                and not interaction.geometry.valid
            ):
                continue

            collected_interactions.append(
                interaction
            )

    return collected_interactions


def group_results_by_pose(
    results: Iterable[SaltBridgeResult],
) -> Dict[str, SaltBridgeResult]:
    """
    Group pose results by normalized pose identifier.

    Parameters
    ----------
    results
        Pose-level results.

    Returns
    -------
    Dict[str, SaltBridgeResult]
        Pose identifier to result mapping.
    """

    grouped_results: Dict[
        str,
        SaltBridgeResult,
    ] = {}

    for result in results:
        if not isinstance(
            result,
            SaltBridgeResult,
        ):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeResult instances."
            )

        pose_key = normalize_grouping_identifier(
            result.pose_id,
            fallback="unassigned_pose",
        )

        if pose_key in grouped_results:
            raise SaltBridgeDetectionError(
                f"Duplicate pose identifier: {pose_key!r}."
            )

        grouped_results[
            pose_key
        ] = result

    return grouped_results


# =============================================================================
# 15.5. INTERACTION PERSISTENCE
# =============================================================================


def multipose_interaction_persistence_key(
    interaction: SaltBridgeInteraction,
    *,
    mode: str = "residue_pair",
) -> Hashable:
    """
    Build a cross-pose persistence key.

    Pose and model identifiers are intentionally excluded.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    mode
        Persistence mode. Supported values are ``"residue_pair"``,
        ``"group_pair"``, ``"atom_pair"``, and ``"group_type"``.

    Returns
    -------
    Hashable
        Cross-pose persistence key.
    """

    normalized_mode = normalize_text(
        mode,
        default="residue_pair",
        lowercase=True,
    )

    if normalized_mode == "residue_pair":
        return interaction_residue_grouping_key(
            interaction,
            directional=True,
        )

    if normalized_mode == "group_pair":
        return interaction_group_pair_grouping_key(
            interaction
        )

    if normalized_mode == "atom_pair":
        return interaction_atom_pair_key(
            interaction,
            include_pose=False,
            include_model=False,
        )

    if normalized_mode == "group_type":
        return (
            normalize_grouping_identifier(
                interaction.cation.group_type,
                fallback="unknown_cation",
            ),
            normalize_grouping_identifier(
                interaction.anion.group_type,
                fallback="unknown_anion",
            ),
        )

    raise SaltBridgeDetectionError(
        f"Unsupported persistence mode: {mode!r}."
    )


def calculate_interaction_persistence(
    results: Iterable[SaltBridgeResult],
    *,
    mode: str = "residue_pair",
    valid_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Calculate interaction persistence across docking poses.

    Persistence is the percentage of analyzed poses containing at least one
    interaction matching the selected key.

    Parameters
    ----------
    results
        Pose-level results.
    mode
        Persistence grouping mode.
    valid_only
        Whether invalid interactions should be ignored.

    Returns
    -------
    List[Dict[str, Any]]
        Persistence records sorted by decreasing pose coverage.
    """

    result_list = list(
        results
    )

    pose_count = len(
        result_list
    )

    if pose_count == 0:
        return []

    persistence_data: Dict[
        Hashable,
        Dict[str, Any],
    ] = {}

    for result in result_list:
        pose_key = normalize_grouping_identifier(
            result.pose_id,
            fallback="unassigned_pose",
        )

        seen_in_current_pose: Set[
            Hashable
        ] = set()

        for interaction in result.interactions:
            if (
                valid_only
                and not interaction.geometry.valid
            ):
                continue

            persistence_key = (
                multipose_interaction_persistence_key(
                    interaction,
                    mode=mode,
                )
            )

            record = persistence_data.setdefault(
                persistence_key,
                {
                    "persistence_key": persistence_key,
                    "pose_ids": set(),
                    "interaction_ids": [],
                    "interaction_count": 0,
                    "scores": [],
                    "distances": [],
                    "strength_counts": {
                        STRENGTH_STRONG: 0,
                        STRENGTH_MODERATE: 0,
                        STRENGTH_WEAK: 0,
                        STRENGTH_REJECTED: 0,
                    },
                },
            )

            record[
                "interaction_count"
            ] += 1

            record[
                "interaction_ids"
            ].append(
                interaction.interaction_id
            )

            score = safe_float(
                interaction.score
            )

            if score is not None:
                record[
                    "scores"
                ].append(
                    score
                )

            interaction_distance = safe_float(
                interaction.distance
            )

            if interaction_distance is not None:
                record[
                    "distances"
                ].append(
                    interaction_distance
                )

            strength = normalize_text(
                interaction.strength,
                default=STRENGTH_REJECTED,
                lowercase=True,
            )

            record[
                "strength_counts"
            ].setdefault(
                strength,
                0,
            )

            record[
                "strength_counts"
            ][
                strength
            ] += 1

            if persistence_key not in seen_in_current_pose:
                record[
                    "pose_ids"
                ].add(
                    pose_key
                )

                seen_in_current_pose.add(
                    persistence_key
                )

    persistence_records: List[
        Dict[str, Any]
    ] = []

    for record in persistence_data.values():
        observed_pose_count = len(
            record["pose_ids"]
        )

        score_statistics = (
            calculate_numeric_statistics(
                record["scores"]
            )
        )

        distance_statistics = (
            calculate_numeric_statistics(
                record["distances"]
            )
        )

        persistence_records.append(
            {
                "persistence_key": (
                    record[
                        "persistence_key"
                    ]
                ),
                "pose_count": (
                    observed_pose_count
                ),
                "total_pose_count": (
                    pose_count
                ),
                "persistence_fraction": (
                    observed_pose_count
                    / pose_count
                ),
                "persistence_percentage": (
                    calculate_percentage(
                        observed_pose_count,
                        pose_count,
                    )
                ),
                "interaction_count": (
                    record[
                        "interaction_count"
                    ]
                ),
                "pose_ids": sorted(
                    record[
                        "pose_ids"
                    ]
                ),
                "interaction_ids": (
                    record[
                        "interaction_ids"
                    ]
                ),
                "score_statistics": (
                    score_statistics
                ),
                "distance_statistics": (
                    distance_statistics
                ),
                "strength_counts": (
                    record[
                        "strength_counts"
                    ]
                ),
            }
        )

    persistence_records.sort(
        key=lambda record: (
            -record[
                "persistence_percentage"
            ],
            -(
                record[
                    "score_statistics"
                ].get(
                    "mean"
                )
                or 0.0
            ),
            (
                record[
                    "distance_statistics"
                ].get(
                    "mean"
                )
                or math.inf
            ),
            repr(
                record[
                    "persistence_key"
                ]
            ),
        )
    )

    for rank, record in enumerate(
        persistence_records,
        start=1,
    ):
        record["rank"] = rank

    return persistence_records


def filter_persistent_interactions(
    persistence_records: Iterable[
        Mapping[str, Any]
    ],
    *,
    minimum_percentage: float = 50.0,
    minimum_pose_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    Filter interaction persistence records.

    Parameters
    ----------
    persistence_records
        Persistence records.
    minimum_percentage
        Minimum persistence percentage.
    minimum_pose_count
        Minimum number of poses.

    Returns
    -------
    List[Dict[str, Any]]
        Filtered persistence records.
    """

    normalized_percentage = safe_float(
        minimum_percentage,
        default=50.0,
    )

    normalized_pose_count = safe_int(
        minimum_pose_count,
        default=1,
    )

    if (
        normalized_percentage is None
        or normalized_percentage < 0.0
        or normalized_percentage > 100.0
    ):
        raise SaltBridgeDetectionError(
            "minimum_percentage must be between 0 and 100."
        )

    if (
        normalized_pose_count is None
        or normalized_pose_count < 1
    ):
        raise SaltBridgeDetectionError(
            "minimum_pose_count must be at least one."
        )

    return [
        dict(record)
        for record in persistence_records
        if (
            safe_float(
                record.get(
                    "persistence_percentage"
                ),
                default=0.0,
            )
            or 0.0
        )
        >= normalized_percentage
        and (
            safe_int(
                record.get(
                    "pose_count"
                ),
                default=0,
            )
            or 0
        )
        >= normalized_pose_count
    ]


# =============================================================================
# 15.6. POSE RANKING
# =============================================================================


def calculate_pose_ranking_score(
    result: SaltBridgeResult,
    *,
    score_weight: float = 1.0,
    interaction_weight: float = 0.25,
    strong_weight: float = 0.50,
    moderate_weight: float = 0.20,
    hotspot_weight: float = 0.10,
) -> float:
    """
    Calculate a salt-bridge-based pose ranking score.

    Parameters
    ----------
    result
        Pose-level salt-bridge result.
    score_weight
        Weight for the total salt-bridge score.
    interaction_weight
        Weight for valid interaction count.
    strong_weight
        Bonus per strong interaction.
    moderate_weight
        Bonus per moderate interaction.
    hotspot_weight
        Bonus per hotspot.

    Returns
    -------
    float
        Pose ranking score.
    """

    compact_summary = (
        result.metadata.get(
            "compact_summary"
        )
        or build_compact_salt_bridge_summary(
            result
        )
    )

    total_score = safe_float(
        compact_summary.get(
            "total_score"
        ),
        default=0.0,
    ) or 0.0

    interaction_count = safe_int(
        compact_summary.get(
            "interaction_count"
        ),
        default=0,
    ) or 0

    strong_count = safe_int(
        compact_summary.get(
            "strong_count"
        ),
        default=0,
    ) or 0

    moderate_count = safe_int(
        compact_summary.get(
            "moderate_count"
        ),
        default=0,
    ) or 0

    hotspot_count = safe_int(
        compact_summary.get(
            "hotspot_count"
        ),
        default=0,
    ) or 0

    return (
        total_score * score_weight
        + interaction_count
        * interaction_weight
        + strong_count
        * strong_weight
        + moderate_count
        * moderate_weight
        + hotspot_count
        * hotspot_weight
    )


def rank_salt_bridge_poses(
    results: Iterable[SaltBridgeResult],
) -> List[Dict[str, Any]]:
    """
    Rank poses using salt-bridge quality metrics.

    Parameters
    ----------
    results
        Pose-level results.

    Returns
    -------
    List[Dict[str, Any]]
        Ranked pose records.
    """

    ranking_records: List[
        Dict[str, Any]
    ] = []

    for result in results:
        if not isinstance(
            result,
            SaltBridgeResult,
        ):
            raise SaltBridgeDetectionError(
                "All values must be SaltBridgeResult instances."
            )

        summary = (
            result.metadata.get(
                "compact_summary"
            )
            or build_compact_salt_bridge_summary(
                result
            )
        )

        ranking_score = (
            calculate_pose_ranking_score(
                result
            )
        )

        ranking_records.append(
            {
                "pose_id": result.pose_id,
                "model_id": result.model_id,
                "ranking_score": (
                    ranking_score
                ),
                "interaction_count": (
                    summary[
                        "interaction_count"
                    ]
                ),
                "total_score": (
                    summary[
                        "total_score"
                    ]
                ),
                "strong_count": (
                    summary[
                        "strong_count"
                    ]
                ),
                "moderate_count": (
                    summary[
                        "moderate_count"
                    ]
                ),
                "weak_count": (
                    summary[
                        "weak_count"
                    ]
                ),
                "minimum_distance": (
                    summary[
                        "minimum_distance"
                    ]
                ),
                "hotspot_count": (
                    summary[
                        "hotspot_count"
                    ]
                ),
                "result": result,
            }
        )

    ranking_records.sort(
        key=lambda record: (
            -record[
                "ranking_score"
            ],
            -record[
                "total_score"
            ],
            -record[
                "strong_count"
            ],
            -record[
                "interaction_count"
            ],
            (
                record[
                    "minimum_distance"
                ]
                if record[
                    "minimum_distance"
                ] is not None
                else math.inf
            ),
            normalize_grouping_identifier(
                record[
                    "pose_id"
                ],
                fallback="unassigned_pose",
            ),
        )
    )

    for rank, record in enumerate(
        ranking_records,
        start=1,
    ):
        record["rank"] = rank

    return ranking_records


def get_best_salt_bridge_pose(
    results: Iterable[SaltBridgeResult],
) -> Optional[SaltBridgeResult]:
    """
    Return the highest-ranked pose result.

    Parameters
    ----------
    results
        Pose-level results.

    Returns
    -------
    Optional[SaltBridgeResult]
        Best result or ``None``.
    """

    ranking = rank_salt_bridge_poses(
        results
    )

    if not ranking:
        return None

    return ranking[0]["result"]


# =============================================================================
# 15.7. MULTIPOSE STATISTICS
# =============================================================================


def calculate_multipose_statistics(
    results: Iterable[SaltBridgeResult],
    *,
    persistence_mode: str = "residue_pair",
) -> Dict[str, Any]:
    """
    Calculate statistics across multiple pose results.

    Parameters
    ----------
    results
        Pose-level salt-bridge results.
    persistence_mode
        Interaction persistence mode.

    Returns
    -------
    Dict[str, Any]
        Multipose statistics.
    """

    result_list = list(
        results
    )

    pose_count = len(
        result_list
    )

    valid_interaction_counts = [
        len(
            result.valid_interactions
        )
        for result in result_list
    ]

    total_scores = [
        get_result_salt_bridge_score(
            result
        )
        for result in result_list
    ]

    strong_counts: List[int] = []
    moderate_counts: List[int] = []
    weak_counts: List[int] = []
    hotspot_counts: List[int] = []
    minimum_distances: List[float] = []

    poses_with_interactions = 0

    for result in result_list:
        summary = (
            result.metadata.get(
                "compact_summary"
            )
            or build_compact_salt_bridge_summary(
                result
            )
        )

        interaction_count = safe_int(
            summary.get(
                "interaction_count"
            ),
            default=0,
        ) or 0

        if interaction_count > 0:
            poses_with_interactions += 1

        strong_counts.append(
            safe_int(
                summary.get(
                    "strong_count"
                ),
                default=0,
            )
            or 0
        )

        moderate_counts.append(
            safe_int(
                summary.get(
                    "moderate_count"
                ),
                default=0,
            )
            or 0
        )

        weak_counts.append(
            safe_int(
                summary.get(
                    "weak_count"
                ),
                default=0,
            )
            or 0
        )

        hotspot_counts.append(
            safe_int(
                summary.get(
                    "hotspot_count"
                ),
                default=0,
            )
            or 0
        )

        minimum_distance = safe_float(
            summary.get(
                "minimum_distance"
            )
        )

        if minimum_distance is not None:
            minimum_distances.append(
                minimum_distance
            )

    persistence_records = (
        calculate_interaction_persistence(
            result_list,
            mode=persistence_mode,
            valid_only=True,
        )
    )

    ranking = rank_salt_bridge_poses(
        result_list
    )

    best_pose_record = (
        ranking[0]
        if ranking
        else None
    )

    return {
        "pose_count": pose_count,
        "successful_pose_count": pose_count,
        "poses_with_interactions": (
            poses_with_interactions
        ),
        "poses_without_interactions": (
            pose_count
            - poses_with_interactions
        ),
        "poses_with_interactions_percentage": (
            calculate_percentage(
                poses_with_interactions,
                pose_count,
            )
        ),
        "total_interaction_count": sum(
            valid_interaction_counts
        ),
        "unique_persistent_interaction_count": (
            len(persistence_records)
        ),
        "total_score": sum(
            total_scores
        ),
        "interactions_per_pose": (
            calculate_numeric_statistics(
                valid_interaction_counts
            )
        ),
        "scores_per_pose": (
            calculate_numeric_statistics(
                total_scores
            )
        ),
        "strong_interactions_per_pose": (
            calculate_numeric_statistics(
                strong_counts
            )
        ),
        "moderate_interactions_per_pose": (
            calculate_numeric_statistics(
                moderate_counts
            )
        ),
        "weak_interactions_per_pose": (
            calculate_numeric_statistics(
                weak_counts
            )
        ),
        "hotspots_per_pose": (
            calculate_numeric_statistics(
                hotspot_counts
            )
        ),
        "minimum_distances_per_pose": (
            calculate_numeric_statistics(
                minimum_distances
            )
        ),
        "best_pose_id": (
            best_pose_record[
                "pose_id"
            ]
            if best_pose_record is not None
            else None
        ),
        "best_model_id": (
            best_pose_record[
                "model_id"
            ]
            if best_pose_record is not None
            else None
        ),
        "best_pose_ranking_score": (
            best_pose_record[
                "ranking_score"
            ]
            if best_pose_record is not None
            else None
        ),
        "best_pose_total_score": (
            best_pose_record[
                "total_score"
            ]
            if best_pose_record is not None
            else None
        ),
        "persistence_mode": (
            persistence_mode
        ),
        "persistence": (
            persistence_records
        ),
        "pose_ranking": ranking,
    }


# =============================================================================
# 15.8. CONSENSUS INTERACTIONS
# =============================================================================


def identify_consensus_salt_bridges(
    results: Iterable[SaltBridgeResult],
    *,
    minimum_persistence_percentage: float = 50.0,
    minimum_pose_count: int = 2,
    mode: str = "residue_pair",
) -> List[Dict[str, Any]]:
    """
    Identify salt bridges conserved across multiple poses.

    Parameters
    ----------
    results
        Pose-level results.
    minimum_persistence_percentage
        Minimum pose coverage percentage.
    minimum_pose_count
        Minimum number of poses.
    mode
        Persistence key mode.

    Returns
    -------
    List[Dict[str, Any]]
        Consensus salt-bridge records.
    """

    persistence_records = (
        calculate_interaction_persistence(
            results,
            mode=mode,
            valid_only=True,
        )
    )

    consensus_records = (
        filter_persistent_interactions(
            persistence_records,
            minimum_percentage=(
                minimum_persistence_percentage
            ),
            minimum_pose_count=(
                minimum_pose_count
            ),
        )
    )

    for record in consensus_records:
        mean_score = (
            record[
                "score_statistics"
            ].get(
                "mean"
            )
            or 0.0
        )

        persistence_fraction = (
            record[
                "persistence_fraction"
            ]
        )

        record[
            "consensus_score"
        ] = (
            persistence_fraction
            * mean_score
        )

    consensus_records.sort(
        key=lambda record: (
            -record[
                "consensus_score"
            ],
            -record[
                "persistence_percentage"
            ],
            -(
                record[
                    "score_statistics"
                ].get(
                    "mean"
                )
                or 0.0
            ),
        )
    )

    for rank, record in enumerate(
        consensus_records,
        start=1,
    ):
        record[
            "consensus_rank"
        ] = rank

    return consensus_records


# =============================================================================
# 15.9. MULTIPOSE SUMMARY
# =============================================================================


def build_multipose_salt_bridge_summary(
    results: Iterable[SaltBridgeResult],
    *,
    persistence_mode: str = "residue_pair",
    consensus_percentage: float = 50.0,
    minimum_consensus_poses: int = 2,
) -> Dict[str, Any]:
    """
    Build a compact summary for multipose salt-bridge analysis.

    Parameters
    ----------
    results
        Pose-level results.
    persistence_mode
        Persistence grouping mode.
    consensus_percentage
        Minimum percentage for consensus interactions.
    minimum_consensus_poses
        Minimum pose count for consensus interactions.

    Returns
    -------
    Dict[str, Any]
        Compact multipose summary.
    """

    result_list = list(
        results
    )

    statistics_data = (
        calculate_multipose_statistics(
            result_list,
            persistence_mode=(
                persistence_mode
            ),
        )
    )

    consensus_interactions = (
        identify_consensus_salt_bridges(
            result_list,
            minimum_persistence_percentage=(
                consensus_percentage
            ),
            minimum_pose_count=(
                minimum_consensus_poses
            ),
            mode=persistence_mode,
        )
    )

    top_consensus = (
        consensus_interactions[0]
        if consensus_interactions
        else None
    )

    return {
        "pose_count": statistics_data[
            "pose_count"
        ],
        "poses_with_interactions": (
            statistics_data[
                "poses_with_interactions"
            ]
        ),
        "poses_without_interactions": (
            statistics_data[
                "poses_without_interactions"
            ]
        ),
        "total_interaction_count": (
            statistics_data[
                "total_interaction_count"
            ]
        ),
        "total_score": (
            statistics_data[
                "total_score"
            ]
        ),
        "mean_interactions_per_pose": (
            statistics_data[
                "interactions_per_pose"
            ].get(
                "mean"
            )
        ),
        "mean_score_per_pose": (
            statistics_data[
                "scores_per_pose"
            ].get(
                "mean"
            )
        ),
        "best_pose_id": (
            statistics_data[
                "best_pose_id"
            ]
        ),
        "best_model_id": (
            statistics_data[
                "best_model_id"
            ]
        ),
        "best_pose_score": (
            statistics_data[
                "best_pose_total_score"
            ]
        ),
        "persistent_interaction_count": (
            statistics_data[
                "unique_persistent_interaction_count"
            ]
        ),
        "consensus_interaction_count": len(
            consensus_interactions
        ),
        "top_consensus_key": (
            top_consensus[
                "persistence_key"
            ]
            if top_consensus is not None
            else None
        ),
        "top_consensus_persistence": (
            top_consensus[
                "persistence_percentage"
            ]
            if top_consensus is not None
            else None
        ),
        "pose_ranking": (
            statistics_data[
                "pose_ranking"
            ]
        ),
        "persistence": (
            statistics_data[
                "persistence"
            ]
        ),
        "consensus_interactions": (
            consensus_interactions
        ),
    }


def build_multipose_text_summary(
    results: Iterable[SaltBridgeResult],
) -> str:
    """
    Build a human-readable multipose summary.

    Parameters
    ----------
    results
        Pose-level results.

    Returns
    -------
    str
        Concise textual summary.
    """

    summary = build_multipose_salt_bridge_summary(
        results
    )

    pose_count = summary[
        "pose_count"
    ]

    if pose_count == 0:
        return (
            "No docking poses were analyzed."
        )

    summary_parts = [
        (
            f"{pose_count} pose"
            f"{'' if pose_count == 1 else 's'} analyzed"
        ),
        (
            f"{summary['poses_with_interactions']} "
            "with valid salt bridges"
        ),
        (
            f"{summary['total_interaction_count']} "
            "total valid interactions"
        ),
        (
            f"total score {summary['total_score']:.3f}"
        ),
    ]

    if summary[
        "best_pose_id"
    ] is not None:
        summary_parts.append(
            f"best pose {summary['best_pose_id']}"
        )

    if summary[
        "consensus_interaction_count"
    ] > 0:
        summary_parts.append(
            f"{summary['consensus_interaction_count']} "
            "consensus interactions"
        )

    return "; ".join(
        summary_parts
    ) + "."


# =============================================================================
# 15.10. COMPLETE MULTIPOSE PIPELINE
# =============================================================================


def analyze_salt_bridges_multipose(
    poses: Iterable[Any],
    config: Optional[SaltBridgeConfig] = None,
    *,
    pose_ids: Optional[
        Union[
            Sequence[Optional[Union[str, int]]],
            Mapping[int, Optional[Union[str, int]]],
        ]
    ] = None,
    model_ids: Optional[
        Union[
            Sequence[Optional[Union[str, int]]],
            Mapping[int, Optional[Union[str, int]]],
        ]
    ] = None,
    attach_to_dock_models: bool = True,
    preserve_existing: bool = False,
    store_full_groups: Optional[bool] = None,
    include_invalid_statistics: bool = False,
    continue_on_error: bool = True,
    persistence_mode: str = "residue_pair",
    consensus_percentage: float = 50.0,
    minimum_consensus_poses: int = 2,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute complete multipose salt-bridge analysis.

    The workflow includes:

    1. pose normalization;
    2. independent analysis of each pose;
    3. pose-level statistics;
    4. interaction persistence analysis;
    5. consensus interaction identification;
    6. pose ranking;
    7. compact and textual multipose summaries.

    Parameters
    ----------
    poses
        Pose-like molecular sources or DockModel-like objects.
    config
        Salt-bridge configuration.
    pose_ids
        Optional pose identifiers.
    model_ids
        Optional model identifiers.
    attach_to_dock_models
        Whether results should be attached to DockModel inputs.
    preserve_existing
        Whether existing DockModel interactions should be preserved.
    store_full_groups
        Whether full group mappings should be retained.
    include_invalid_statistics
        Whether rejected candidates should enter descriptive statistics.
    continue_on_error
        Whether processing should continue after pose-level failures.
    persistence_mode
        Cross-pose persistence mode.
    consensus_percentage
        Minimum persistence percentage for consensus interactions.
    minimum_consensus_poses
        Minimum number of poses for consensus interactions.
    warnings
        Optional warning collector.

    Returns
    -------
    Dict[str, Any]
        Complete multipose analysis result.
    """

    warning_list = (
        warnings
        if warnings is not None
        else []
    )

    pose_list = normalize_pose_collection(
        poses
    )

    results = analyze_multiple_poses_salt_bridges(
        pose_list,
        config,
        pose_ids=pose_ids,
        model_ids=model_ids,
        attach_to_dock_models=(
            attach_to_dock_models
        ),
        preserve_existing=(
            preserve_existing
        ),
        store_full_groups=(
            store_full_groups
        ),
        include_invalid_statistics=(
            include_invalid_statistics
        ),
        continue_on_error=(
            continue_on_error
        ),
        warnings=warning_list,
    )

    multipose_statistics = (
        calculate_multipose_statistics(
            results,
            persistence_mode=(
                persistence_mode
            ),
        )
    )

    consensus_interactions = (
        identify_consensus_salt_bridges(
            results,
            minimum_persistence_percentage=(
                consensus_percentage
            ),
            minimum_pose_count=(
                minimum_consensus_poses
            ),
            mode=persistence_mode,
        )
    )

    compact_summary = (
        build_multipose_salt_bridge_summary(
            results,
            persistence_mode=(
                persistence_mode
            ),
            consensus_percentage=(
                consensus_percentage
            ),
            minimum_consensus_poses=(
                minimum_consensus_poses
            ),
        )
    )

    text_summary = (
        build_multipose_text_summary(
            results
        )
    )

    return {
        "results": results,
        "interactions": (
            collect_multipose_interactions(
                results,
                valid_only=True,
            )
        ),
        "statistics": (
            multipose_statistics
        ),
        "pose_ranking": (
            multipose_statistics[
                "pose_ranking"
            ]
        ),
        "persistence": (
            multipose_statistics[
                "persistence"
            ]
        ),
        "consensus_interactions": (
            consensus_interactions
        ),
        "compact_summary": (
            compact_summary
        ),
        "text_summary": (
            text_summary
        ),
        "warnings": warning_list,
        "metadata": {
            "input_pose_count": len(
                pose_list
            ),
            "successful_pose_count": len(
                results
            ),
            "failed_pose_count": (
                len(pose_list)
                - len(results)
            ),
            "persistence_mode": (
                persistence_mode
            ),
            "consensus_percentage": (
                consensus_percentage
            ),
            "minimum_consensus_poses": (
                minimum_consensus_poses
            ),
            "multipose_analysis_completed": (
                True
            ),
        },
    }


# =============================================================================
# 16. SERIALIZATION AND EXPORT PREPARATION
# =============================================================================


# =============================================================================
# 16.1. SERIALIZATION UTILITIES
# =============================================================================


def is_json_primitive(
    value: Any,
) -> bool:
    """
    Return whether a value is directly JSON serializable.

    Parameters
    ----------
    value
        Value to inspect.

    Returns
    -------
    bool
        Whether the value is a JSON primitive.
    """

    return (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
    )


def sanitize_json_number(
    value: Any,
) -> Optional[Union[int, float]]:
    """
    Convert a numeric-like value into a JSON-safe number.

    Non-finite floating-point values are converted to ``None``.

    Parameters
    ----------
    value
        Numeric-like value.

    Returns
    -------
    Optional[Union[int, float]]
        JSON-safe numeric value.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    normalized_value = safe_float(
        value
    )

    if (
        normalized_value is None
        or not math.isfinite(normalized_value)
    ):
        return None

    return normalized_value


def sanitize_json_key(
    value: Any,
) -> str:
    """
    Convert a mapping key into a stable JSON string key.

    Parameters
    ----------
    value
        Mapping key.

    Returns
    -------
    str
        JSON-safe key.
    """

    if value is None:
        return "null"

    if isinstance(value, tuple):
        return "|".join(
            sanitize_json_key(
                item
            )
            for item in value
        )

    if isinstance(value, list):
        return "|".join(
            sanitize_json_key(
                item
            )
            for item in value
        )

    if isinstance(value, set):
        return "|".join(
            sorted(
                sanitize_json_key(
                    item
                )
                for item in value
            )
        )

    return str(value)


def make_json_safe(
    value: Any,
    *,
    max_depth: int = 20,
    current_depth: int = 0,
    fallback_to_string: bool = True,
) -> Any:
    """
    Recursively convert a value into JSON-safe data.

    Parameters
    ----------
    value
        Value to convert.
    max_depth
        Maximum recursion depth.
    current_depth
        Current recursion depth.
    fallback_to_string
        Whether unsupported objects should be converted to strings.

    Returns
    -------
    Any
        JSON-compatible representation.

    Raises
    ------
    SaltBridgeSerializationError
        If maximum depth is exceeded or an unsupported value cannot be handled.
    """

    if current_depth > max_depth:
        raise SaltBridgeSerializationError(
            "Maximum serialization depth exceeded."
        )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(value, Mapping):
        return {
            sanitize_json_key(
                key
            ): make_json_safe(
                item,
                max_depth=max_depth,
                current_depth=(
                    current_depth + 1
                ),
                fallback_to_string=(
                    fallback_to_string
                ),
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
        iterable_value = (
            sorted(
                value,
                key=repr,
            )
            if isinstance(
                value,
                (
                    set,
                    frozenset,
                ),
            )
            else value
        )

        return [
            make_json_safe(
                item,
                max_depth=max_depth,
                current_depth=(
                    current_depth + 1
                ),
                fallback_to_string=(
                    fallback_to_string
                ),
            )
            for item in iterable_value
        ]

    if hasattr(
        value,
        "tolist",
    ):
        try:
            return make_json_safe(
                value.tolist(),
                max_depth=max_depth,
                current_depth=(
                    current_depth + 1
                ),
                fallback_to_string=(
                    fallback_to_string
                ),
            )

        except Exception:
            pass

    if fallback_to_string:
        return str(value)

    raise SaltBridgeSerializationError(
        "Unsupported value encountered during JSON serialization: "
        f"{type(value).__name__}."
    )


def serialize_coordinate(
    coordinate: Optional[
        Sequence[float]
    ],
) -> Optional[List[float]]:
    """
    Convert a coordinate into a JSON-safe three-value list.

    Parameters
    ----------
    coordinate
        Coordinate-like value.

    Returns
    -------
    Optional[List[float]]
        Serialized coordinate or ``None``.
    """

    if coordinate is None:
        return None

    try:
        normalized_coordinate = (
            normalize_coordinate(
                coordinate
            )
        )

    except SaltBridgeGeometryError:
        return None

    return [
        float(
            normalized_coordinate[0]
        ),
        float(
            normalized_coordinate[1]
        ),
        float(
            normalized_coordinate[2]
        ),
    ]


# =============================================================================
# 16.2. ATOM AND RESIDUE SERIALIZATION
# =============================================================================


def atom_reference_to_dict(
    atom: Any,
    *,
    include_coordinate: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build a compact serializable atom reference.

    The molecular object itself is not serialized.

    Parameters
    ----------
    atom
        Atom-like object.
    include_coordinate
        Whether atom coordinates should be included.

    Returns
    -------
    Optional[Dict[str, Any]]
        Atom reference or ``None``.
    """

    if atom is None:
        return None

    residue = get_atom_residue(
        atom
    )

    atom_data: Dict[str, Any] = {
        "name": get_atom_name(
            atom
        ),
        "element": get_atom_element(
            atom
        ),
        "serial": get_atom_serial(
            atom
        ),
        "atom_label": make_atom_label(
            atom
        ),
        "atom_identity": make_json_safe(
            atom_identity(
                atom
            )
        ),
        "residue": (
            residue_reference_to_dict(
                residue
            )
            if residue is not None
            else None
        ),
    }

    if include_coordinate:
        atom_data[
            "coordinate"
        ] = serialize_coordinate(
            get_atom_coordinate(
                atom,
                required=False,
            )
        )

    return atom_data


def residue_reference_to_dict(
    residue: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build a compact serializable residue reference.

    Parameters
    ----------
    residue
        Residue-like object.

    Returns
    -------
    Optional[Dict[str, Any]]
        Residue reference or ``None``.
    """

    if residue is None:
        return None

    return {
        "name": get_residue_name(
            residue
        ),
        "number": get_residue_number(
            residue
        ),
        "chain_id": get_chain_id(
            residue
        ),
        "label": make_residue_label(
            residue,
            fallback="unknown_residue",
        ),
        "identity": make_json_safe(
            residue_identity(
                residue
            )
        ),
    }


# =============================================================================
# 16.3. CHARGED ATOM SERIALIZATION
# =============================================================================


def charged_atom_to_dict(
    charged_atom: ChargedAtom,
    *,
    include_atom_reference: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Convert a ChargedAtom into a serializable dictionary.

    Parameters
    ----------
    charged_atom
        Charged atom.
    include_atom_reference
        Whether the underlying atom reference should be included.
    include_metadata
        Whether metadata should be included.

    Returns
    -------
    Dict[str, Any]
        Serialized charged atom.
    """

    if not isinstance(
        charged_atom,
        ChargedAtom,
    ):
        raise SaltBridgeSerializationError(
            "charged_atom must be a ChargedAtom instance."
        )

    atom_data: Dict[str, Any] = {
        "name": charged_atom.name,
        "element": charged_atom.element,
        "coordinate": serialize_coordinate(
            charged_atom.coordinate
        ),
        "formal_charge": sanitize_json_number(
            charged_atom.formal_charge
        ),
        "partial_charge": sanitize_json_number(
            charged_atom.partial_charge
        ),
        "effective_charge": sanitize_json_number(
            charged_atom.effective_charge
        ),
        "polarity": charged_atom.polarity,
        "source": charged_atom.source,
        "has_coordinates": (
            charged_atom.has_coordinates
        ),
        "is_positive": (
            charged_atom.is_positive
        ),
        "is_negative": (
            charged_atom.is_negative
        ),
        "residue": residue_reference_to_dict(
            charged_atom.residue
        ),
    }

    if include_atom_reference:
        atom_data[
            "atom"
        ] = atom_reference_to_dict(
            charged_atom.atom,
            include_coordinate=False,
        )

    if include_metadata:
        atom_data[
            "metadata"
        ] = make_json_safe(
            charged_atom.metadata
        )

    return atom_data


# =============================================================================
# 16.4. CHARGED GROUP SERIALIZATION
# =============================================================================


def charged_group_to_dict(
    group: ChargedGroup,
    *,
    include_atoms: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Convert a ChargedGroup into a serializable dictionary.

    Parameters
    ----------
    group
        Charged group.
    include_atoms
        Whether charged atoms should be included.
    include_metadata
        Whether group metadata should be included.

    Returns
    -------
    Dict[str, Any]
        Serialized charged group.
    """

    if not isinstance(
        group,
        ChargedGroup,
    ):
        raise SaltBridgeSerializationError(
            "group must be a ChargedGroup instance."
        )

    group_data: Dict[str, Any] = {
        "group_id": group.group_id,
        "group_type": group.group_type,
        "polarity": group.polarity,
        "center": serialize_coordinate(
            group.center
        ),
        "net_charge": sanitize_json_number(
            group.net_charge
        ),
        "source": group.source,
        "confidence": sanitize_json_number(
            group.confidence
        ),
        "residue": residue_reference_to_dict(
            group.residue
        ),
        "representative_atom": (
            atom_reference_to_dict(
                group.representative_atom
            )
        ),
        "atom_count": len(
            group.atoms
        ),
        "label": make_group_label(
            group
        ),
        "identity": make_json_safe(
            charged_group_identity(
                group
            )
        ),
    }

    if include_atoms:
        group_data["atoms"] = [
            charged_atom_to_dict(
                charged_atom,
                include_atom_reference=True,
                include_metadata=(
                    include_metadata
                ),
            )
            for charged_atom in group.atoms
        ]

    if include_metadata:
        group_data[
            "metadata"
        ] = make_json_safe(
            group.metadata
        )

    return group_data


# =============================================================================
# 16.5. GEOMETRY SERIALIZATION
# =============================================================================


def salt_bridge_geometry_to_dict(
    geometry: SaltBridgeGeometry,
) -> Dict[str, Any]:
    """
    Convert SaltBridgeGeometry into a serializable dictionary.

    Parameters
    ----------
    geometry
        Salt-bridge geometry.

    Returns
    -------
    Dict[str, Any]
        Serialized geometry.
    """

    if not isinstance(
        geometry,
        SaltBridgeGeometry,
    ):
        raise SaltBridgeSerializationError(
            "geometry must be a SaltBridgeGeometry instance."
        )

    return {
        "center_distance": sanitize_json_number(
            geometry.center_distance
        ),
        "minimum_atom_distance": sanitize_json_number(
            geometry.minimum_atom_distance
        ),
        "maximum_atom_distance": sanitize_json_number(
            geometry.maximum_atom_distance
        ),
        "mean_atom_distance": sanitize_json_number(
            geometry.mean_atom_distance
        ),
        "contact_count": safe_int(
            geometry.contact_count,
            default=0,
        ),
        "closest_positive_atom": (
            atom_reference_to_dict(
                geometry.closest_positive_atom
            )
        ),
        "closest_negative_atom": (
            atom_reference_to_dict(
                geometry.closest_negative_atom
            )
        ),
        "valid": bool(
            geometry.valid
        ),
        "rejection_reason": (
            geometry.rejection_reason
        ),
    }


# =============================================================================
# 16.6. INTERACTION SERIALIZATION
# =============================================================================


def salt_bridge_interaction_to_dict(
    interaction: SaltBridgeInteraction,
    *,
    include_groups: bool = True,
    include_group_atoms: bool = True,
    include_metadata: bool = True,
    compact: bool = False,
) -> Dict[str, Any]:
    """
    Convert a SaltBridgeInteraction into a serializable dictionary.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    include_groups
        Whether full charged-group data should be included.
    include_group_atoms
        Whether charged-group atoms should be included.
    include_metadata
        Whether interaction metadata should be included.
    compact
        Whether a compact representation should be generated.

    Returns
    -------
    Dict[str, Any]
        Serialized interaction.
    """

    if not isinstance(
        interaction,
        SaltBridgeInteraction,
    ):
        raise SaltBridgeSerializationError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    if compact:
        return make_json_safe(
            build_interaction_summary_record(
                interaction
            )
        )

    interaction_data: Dict[str, Any] = {
        "interaction_id": (
            interaction.interaction_id
        ),
        "interaction_type": (
            interaction.interaction_type
        ),
        "strength": interaction.strength,
        "score": sanitize_json_number(
            interaction.score
        ),
        "pose_id": make_json_safe(
            interaction.pose_id
        ),
        "model_id": make_json_safe(
            interaction.model_id
        ),
        "geometry": (
            salt_bridge_geometry_to_dict(
                interaction.geometry
            )
        ),
    }

    if include_groups:
        interaction_data["cation"] = (
            charged_group_to_dict(
                interaction.cation,
                include_atoms=(
                    include_group_atoms
                ),
                include_metadata=(
                    include_metadata
                ),
            )
        )

        interaction_data["anion"] = (
            charged_group_to_dict(
                interaction.anion,
                include_atoms=(
                    include_group_atoms
                ),
                include_metadata=(
                    include_metadata
                ),
            )
        )

    else:
        interaction_data[
            "cation"
        ] = {
            "group_id": (
                interaction.cation.group_id
            ),
            "group_type": (
                interaction.cation.group_type
            ),
            "label": make_group_label(
                interaction.cation
            ),
            "residue": (
                residue_reference_to_dict(
                    interaction.cation.residue
                )
            ),
        }

        interaction_data[
            "anion"
        ] = {
            "group_id": (
                interaction.anion.group_id
            ),
            "group_type": (
                interaction.anion.group_type
            ),
            "label": make_group_label(
                interaction.anion
            ),
            "residue": (
                residue_reference_to_dict(
                    interaction.anion.residue
                )
            ),
        }

    if include_metadata:
        interaction_data[
            "metadata"
        ] = make_json_safe(
            interaction.metadata
        )

    return interaction_data


# =============================================================================
# 16.7. RESULT SERIALIZATION
# =============================================================================


def salt_bridge_result_to_dict(
    result: SaltBridgeResult,
    *,
    include_interactions: bool = True,
    include_groups: bool = True,
    include_group_atoms: bool = True,
    include_statistics: bool = True,
    include_metadata: bool = True,
    include_warnings: bool = True,
    compact_interactions: bool = False,
) -> Dict[str, Any]:
    """
    Convert a SaltBridgeResult into a serializable dictionary.

    Parameters
    ----------
    result
        Salt-bridge result.
    include_interactions
        Whether interactions should be included.
    include_groups
        Whether recognized charged groups should be included.
    include_group_atoms
        Whether atoms within recognized groups should be included.
    include_statistics
        Whether statistics should be included.
    include_metadata
        Whether metadata should be included.
    include_warnings
        Whether warnings should be included.
    compact_interactions
        Whether interactions should use flat summary records.

    Returns
    -------
    Dict[str, Any]
        Serialized result.
    """

    if not isinstance(
        result,
        SaltBridgeResult,
    ):
        raise SaltBridgeSerializationError(
            "result must be a SaltBridgeResult instance."
        )

    result_data: Dict[str, Any] = {
        "schema": "dockanalyzer.saltbridge",
        "schema_version": "1.0",
        "module_version": __version__,
        "pose_id": make_json_safe(
            result.pose_id
        ),
        "model_id": make_json_safe(
            result.model_id
        ),
        "interaction_count": len(
            result.interactions
        ),
        "cationic_group_count": len(
            result.cationic_groups
        ),
        "anionic_group_count": len(
            result.anionic_groups
        ),
    }

    if include_interactions:
        result_data[
            "interactions"
        ] = [
            salt_bridge_interaction_to_dict(
                interaction,
                include_groups=(
                    not compact_interactions
                ),
                include_group_atoms=(
                    include_group_atoms
                ),
                include_metadata=(
                    include_metadata
                ),
                compact=(
                    compact_interactions
                ),
            )
            for interaction in result.interactions
        ]

    if include_groups:
        result_data[
            "cationic_groups"
        ] = [
            charged_group_to_dict(
                group,
                include_atoms=(
                    include_group_atoms
                ),
                include_metadata=(
                    include_metadata
                ),
            )
            for group in result.cationic_groups
        ]

        result_data[
            "anionic_groups"
        ] = [
            charged_group_to_dict(
                group,
                include_atoms=(
                    include_group_atoms
                ),
                include_metadata=(
                    include_metadata
                ),
            )
            for group in result.anionic_groups
        ]

    if include_statistics:
        result_data[
            "statistics"
        ] = make_json_safe(
            result.statistics
        )

    if include_warnings:
        result_data[
            "warnings"
        ] = [
            str(warning)
            for warning in result.warnings
        ]

    if include_metadata:
        result_data[
            "metadata"
        ] = make_json_safe(
            result.metadata
        )

    return make_json_safe(
        result_data
    )


# =============================================================================
# 16.8. JSON SERIALIZATION
# =============================================================================


def serialize_salt_bridge_result(
    result: SaltBridgeResult,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    compact_interactions: bool = False,
    include_groups: bool = True,
    include_group_atoms: bool = True,
    include_statistics: bool = True,
    include_metadata: bool = True,
) -> str:
    """
    Serialize a SaltBridgeResult to JSON.

    Parameters
    ----------
    result
        Salt-bridge result.
    indent
        JSON indentation level.
    sort_keys
        Whether JSON keys should be sorted.
    ensure_ascii
        Whether non-ASCII characters should be escaped.
    compact_interactions
        Whether flat interaction records should be used.
    include_groups
        Whether recognized charged groups should be included.
    include_group_atoms
        Whether group atoms should be included.
    include_statistics
        Whether statistics should be included.
    include_metadata
        Whether metadata should be included.

    Returns
    -------
    str
        JSON document.
    """

    try:
        payload = salt_bridge_result_to_dict(
            result,
            include_interactions=True,
            include_groups=include_groups,
            include_group_atoms=(
                include_group_atoms
            ),
            include_statistics=(
                include_statistics
            ),
            include_metadata=(
                include_metadata
            ),
            include_warnings=True,
            compact_interactions=(
                compact_interactions
            ),
        )

        return json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )

    except SaltBridgeSerializationError:
        raise

    except (
        TypeError,
        ValueError,
    ) as error:
        raise SaltBridgeSerializationError(
            "Could not serialize SaltBridgeResult to JSON."
        ) from error


def serialize_salt_bridge_interactions(
    interactions: Iterable[
        SaltBridgeInteraction
    ],
    *,
    indent: Optional[int] = 2,
    compact: bool = False,
    include_metadata: bool = True,
) -> str:
    """
    Serialize salt-bridge interactions to JSON.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    indent
        JSON indentation level.
    compact
        Whether flat summary records should be used.
    include_metadata
        Whether interaction metadata should be included.

    Returns
    -------
    str
        JSON document.
    """

    try:
        payload = [
            salt_bridge_interaction_to_dict(
                interaction,
                include_groups=not compact,
                include_group_atoms=not compact,
                include_metadata=(
                    include_metadata
                ),
                compact=compact,
            )
            for interaction in interactions
        ]

        return json.dumps(
            make_json_safe(
                payload
            ),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    except SaltBridgeSerializationError:
        raise

    except (
        TypeError,
        ValueError,
    ) as error:
        raise SaltBridgeSerializationError(
            "Could not serialize salt-bridge interactions."
        ) from error


# =============================================================================
# 16.9. TABLE EXPORT RECORDS
# =============================================================================


def salt_bridge_interactions_to_rows(
    interactions: Iterable[
        SaltBridgeInteraction
    ],
    *,
    include_invalid: bool = False,
    sort_by_score: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert interactions into flat tabular rows.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    include_invalid
        Whether invalid interactions should be included.
    sort_by_score
        Whether rows should be ordered by decreasing score.

    Returns
    -------
    List[Dict[str, Any]]
        Flat table rows.
    """

    rows = build_interaction_summary_table(
        interactions,
        include_invalid=(
            include_invalid
        ),
        sort_by_score=sort_by_score,
    )

    return [
        make_json_safe(
            row
        )
        for row in rows
    ]


def salt_bridge_groups_to_rows(
    groups: Iterable[ChargedGroup],
) -> List[Dict[str, Any]]:
    """
    Convert charged groups into flat tabular rows.

    Parameters
    ----------
    groups
        Charged groups.

    Returns
    -------
    List[Dict[str, Any]]
        Flat group rows.
    """

    rows: List[Dict[str, Any]] = []

    for group in groups:
        if not isinstance(
            group,
            ChargedGroup,
        ):
            raise SaltBridgeSerializationError(
                "All values must be ChargedGroup instances."
            )

        residue_data = (
            residue_reference_to_dict(
                group.residue
            )
            or {}
        )

        center = serialize_coordinate(
            group.center
        )

        rows.append(
            {
                "group_id": group.group_id,
                "group_type": (
                    group.group_type
                ),
                "polarity": group.polarity,
                "net_charge": (
                    sanitize_json_number(
                        group.net_charge
                    )
                ),
                "confidence": (
                    sanitize_json_number(
                        group.confidence
                    )
                ),
                "source": group.source,
                "atom_count": len(
                    group.atoms
                ),
                "residue_label": (
                    residue_data.get(
                        "label"
                    )
                ),
                "residue_name": (
                    residue_data.get(
                        "name"
                    )
                ),
                "residue_number": (
                    residue_data.get(
                        "number"
                    )
                ),
                "chain_id": (
                    residue_data.get(
                        "chain_id"
                    )
                ),
                "center_x": (
                    center[0]
                    if center is not None
                    else None
                ),
                "center_y": (
                    center[1]
                    if center is not None
                    else None
                ),
                "center_z": (
                    center[2]
                    if center is not None
                    else None
                ),
            }
        )

    return rows


def salt_bridge_statistics_to_rows(
    statistics_data: Mapping[str, Any],
    *,
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    Flatten nested statistics into metric-value rows.

    Parameters
    ----------
    statistics_data
        Statistics mapping.
    prefix
        Optional metric prefix.

    Returns
    -------
    List[Dict[str, Any]]
        Flattened statistics rows.
    """

    rows: List[Dict[str, Any]] = []

    def walk(
        value: Any,
        path: str,
    ) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = sanitize_json_key(
                    key
                )

                nested_path = (
                    f"{path}.{normalized_key}"
                    if path
                    else normalized_key
                )

                walk(
                    nested_value,
                    nested_path,
                )

            return

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            rows.append(
                {
                    "metric": path,
                    "value": make_json_safe(
                        value
                    ),
                }
            )

            return

        rows.append(
            {
                "metric": path,
                "value": make_json_safe(
                    value
                ),
            }
        )

    walk(
        statistics_data,
        prefix,
    )

    return rows


# =============================================================================
# 16.10. RESIDUE SUMMARY EXPORT
# =============================================================================


def build_residue_salt_bridge_summary_rows(
    interactions: Iterable[
        SaltBridgeInteraction
    ],
) -> List[Dict[str, Any]]:
    """
    Build residue-level salt-bridge summary rows.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.

    Returns
    -------
    List[Dict[str, Any]]
        Residue summary rows.
    """

    residue_groups = (
        group_salt_bridges_by_any_residue(
            interactions
        )
    )

    hotspot_records = (
        identify_residue_hotspots(
            interactions,
            include_singletons=True,
        )
    )

    hotspot_by_key = {
        sanitize_json_key(
            record.get(
                "residue_key"
            )
        ): record
        for record in hotspot_records
    }

    rows: List[Dict[str, Any]] = []

    for residue_key, residue_interactions in (
        residue_groups.items()
    ):
        group_summary = (
            summarize_interaction_group(
                residue_interactions
            )
        )

        normalized_key = sanitize_json_key(
            residue_key
        )

        hotspot = hotspot_by_key.get(
            normalized_key,
            {},
        )

        cation_count = sum(
            1
            for interaction
            in residue_interactions
            if residue_identity(
                interaction.cation.residue
            )
            == residue_key
        )

        anion_count = sum(
            1
            for interaction
            in residue_interactions
            if residue_identity(
                interaction.anion.residue
            )
            == residue_key
        )

        rows.append(
            {
                "residue_key": (
                    normalized_key
                ),
                "residue_label": (
                    hotspot.get(
                        "residue_label",
                        normalized_key,
                    )
                ),
                "interaction_count": (
                    group_summary.get(
                        "interaction_count",
                        0,
                    )
                ),
                "cation_interaction_count": (
                    cation_count
                ),
                "anion_interaction_count": (
                    anion_count
                ),
                "total_score": (
                    group_summary.get(
                        "total_score",
                        0.0,
                    )
                ),
                "mean_score": (
                    group_summary.get(
                        "mean_score"
                    )
                ),
                "minimum_distance": (
                    group_summary.get(
                        "minimum_distance"
                    )
                ),
                "mean_distance": (
                    group_summary.get(
                        "mean_distance"
                    )
                ),
                "strong_count": (
                    group_summary
                    .get(
                        "strength_counts",
                        {},
                    )
                    .get(
                        STRENGTH_STRONG,
                        0,
                    )
                ),
                "moderate_count": (
                    group_summary
                    .get(
                        "strength_counts",
                        {},
                    )
                    .get(
                        STRENGTH_MODERATE,
                        0,
                    )
                ),
                "weak_count": (
                    group_summary
                    .get(
                        "strength_counts",
                        {},
                    )
                    .get(
                        STRENGTH_WEAK,
                        0,
                    )
                ),
                "hotspot_score": (
                    hotspot.get(
                        "hotspot_score"
                    )
                ),
                "hotspot_rank": (
                    hotspot.get(
                        "rank"
                    )
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -(
                safe_float(
                    row.get(
                        "hotspot_score"
                    ),
                    default=0.0,
                )
                or 0.0
            ),
            -(
                safe_float(
                    row.get(
                        "total_score"
                    ),
                    default=0.0,
                )
                or 0.0
            ),
            str(
                row.get(
                    "residue_label",
                    "",
                )
            ),
        )
    )

    return rows


# =============================================================================
# 16.11. POSE SUMMARY EXPORT
# =============================================================================


def build_pose_salt_bridge_summary_rows(
    results: Iterable[
        SaltBridgeResult
    ],
) -> List[Dict[str, Any]]:
    """
    Build pose-level summary rows.

    Parameters
    ----------
    results
        Pose-level results.

    Returns
    -------
    List[Dict[str, Any]]
        Pose summary rows.
    """

    ranking = rank_salt_bridge_poses(
        results
    )

    rows: List[Dict[str, Any]] = []

    for ranking_record in ranking:
        result = ranking_record[
            "result"
        ]

        summary = (
            result.metadata.get(
                "compact_summary"
            )
            or build_compact_salt_bridge_summary(
                result
            )
        )

        rows.append(
            {
                "rank": ranking_record[
                    "rank"
                ],
                "pose_id": result.pose_id,
                "model_id": result.model_id,
                "ranking_score": (
                    ranking_record[
                        "ranking_score"
                    ]
                ),
                "interaction_count": (
                    summary[
                        "interaction_count"
                    ]
                ),
                "total_score": (
                    summary[
                        "total_score"
                    ]
                ),
                "mean_score": (
                    summary[
                        "mean_score"
                    ]
                ),
                "best_score": (
                    summary[
                        "best_score"
                    ]
                ),
                "minimum_distance": (
                    summary[
                        "minimum_distance"
                    ]
                ),
                "mean_distance": (
                    summary[
                        "mean_distance"
                    ]
                ),
                "strong_count": (
                    summary[
                        "strong_count"
                    ]
                ),
                "moderate_count": (
                    summary[
                        "moderate_count"
                    ]
                ),
                "weak_count": (
                    summary[
                        "weak_count"
                    ]
                ),
                "residue_count": (
                    summary[
                        "residue_count"
                    ]
                ),
                "hotspot_count": (
                    summary[
                        "hotspot_count"
                    ]
                ),
                "intrachain_count": (
                    summary[
                        "intrachain_count"
                    ]
                ),
                "interchain_count": (
                    summary[
                        "interchain_count"
                    ]
                ),
            }
        )

    return make_json_safe(
        rows
    )


# =============================================================================
# 16.12. MULTIPOSE SERIALIZATION
# =============================================================================


def salt_bridge_multipose_to_dict(
    multipose_result: Mapping[str, Any],
    *,
    include_pose_results: bool = True,
    include_pose_interactions: bool = True,
    compact_pose_interactions: bool = True,
    include_statistics: bool = True,
    include_persistence: bool = True,
    include_consensus: bool = True,
) -> Dict[str, Any]:
    """
    Convert a complete multipose result into serializable data.

    Parameters
    ----------
    multipose_result
        Result returned by ``analyze_salt_bridges_multipose``.
    include_pose_results
        Whether individual pose results should be included.
    include_pose_interactions
        Whether pose interactions should be included.
    compact_pose_interactions
        Whether pose interactions should use flat records.
    include_statistics
        Whether multipose statistics should be included.
    include_persistence
        Whether persistence records should be included.
    include_consensus
        Whether consensus records should be included.

    Returns
    -------
    Dict[str, Any]
        Serialized multipose result.
    """

    if not isinstance(
        multipose_result,
        Mapping,
    ):
        raise SaltBridgeSerializationError(
            "multipose_result must be a mapping."
        )

    payload: Dict[str, Any] = {
        "schema": (
            "dockanalyzer.saltbridge.multipose"
        ),
        "schema_version": "1.0",
        "module_version": __version__,
        "compact_summary": make_json_safe(
            multipose_result.get(
                "compact_summary",
                {},
            )
        ),
        "text_summary": str(
            multipose_result.get(
                "text_summary",
                "",
            )
        ),
        "warnings": make_json_safe(
            multipose_result.get(
                "warnings",
                [],
            )
        ),
        "metadata": make_json_safe(
            multipose_result.get(
                "metadata",
                {},
            )
        ),
    }

    if include_pose_results:
        pose_results = (
            multipose_result.get(
                "results",
                [],
            )
        )

        payload[
            "results"
        ] = [
            salt_bridge_result_to_dict(
                result,
                include_interactions=(
                    include_pose_interactions
                ),
                include_groups=False,
                include_group_atoms=False,
                include_statistics=True,
                include_metadata=True,
                include_warnings=True,
                compact_interactions=(
                    compact_pose_interactions
                ),
            )
            for result in pose_results
            if isinstance(
                result,
                SaltBridgeResult,
            )
        ]

    if include_statistics:
        payload[
            "statistics"
        ] = make_json_safe(
            multipose_result.get(
                "statistics",
                {},
            )
        )

    if include_persistence:
        payload[
            "persistence"
        ] = make_json_safe(
            multipose_result.get(
                "persistence",
                [],
            )
        )

    if include_consensus:
        payload[
            "consensus_interactions"
        ] = make_json_safe(
            multipose_result.get(
                "consensus_interactions",
                [],
            )
        )

    payload[
        "pose_ranking"
    ] = make_json_safe(
        [
            {
                key: value
                for key, value
                in record.items()
                if key != "result"
            }
            for record in multipose_result.get(
                "pose_ranking",
                [],
            )
        ]
    )

    return make_json_safe(
        payload
    )


def serialize_salt_bridge_multipose(
    multipose_result: Mapping[str, Any],
    *,
    indent: Optional[int] = 2,
    include_pose_results: bool = True,
    compact_pose_interactions: bool = True,
) -> str:
    """
    Serialize complete multipose salt-bridge analysis to JSON.

    Parameters
    ----------
    multipose_result
        Complete multipose analysis.
    indent
        JSON indentation level.
    include_pose_results
        Whether individual results should be included.
    compact_pose_interactions
        Whether individual interactions should be compact.

    Returns
    -------
    str
        JSON document.
    """

    try:
        payload = salt_bridge_multipose_to_dict(
            multipose_result,
            include_pose_results=(
                include_pose_results
            ),
            include_pose_interactions=True,
            compact_pose_interactions=(
                compact_pose_interactions
            ),
            include_statistics=True,
            include_persistence=True,
            include_consensus=True,
        )

        return json.dumps(
            payload,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    except SaltBridgeSerializationError:
        raise

    except (
        TypeError,
        ValueError,
    ) as error:
        raise SaltBridgeSerializationError(
            "Could not serialize multipose salt-bridge analysis."
        ) from error


# =============================================================================
# 16.13. EXPORT PAYLOAD ASSEMBLY
# =============================================================================


def build_salt_bridge_export_payload(
    result: SaltBridgeResult,
    *,
    include_full_result: bool = True,
    include_interaction_table: bool = True,
    include_group_tables: bool = True,
    include_residue_summary: bool = True,
    include_statistics_table: bool = True,
) -> Dict[str, Any]:
    """
    Build a complete export payload for one SaltBridgeResult.

    This function prepares data but does not write files.

    Parameters
    ----------
    result
        Salt-bridge result.
    include_full_result
        Whether the complete serialized result should be included.
    include_interaction_table
        Whether a flat interaction table should be included.
    include_group_tables
        Whether cationic and anionic group tables should be included.
    include_residue_summary
        Whether residue summary rows should be included.
    include_statistics_table
        Whether flattened statistics should be included.

    Returns
    -------
    Dict[str, Any]
        Export payload.
    """

    if not isinstance(
        result,
        SaltBridgeResult,
    ):
        raise SaltBridgeSerializationError(
            "result must be a SaltBridgeResult instance."
        )

    if not result.statistics:
        calculate_salt_bridge_result_statistics(
            result,
            in_place=True,
        )

    payload: Dict[str, Any] = {
        "schema": (
            "dockanalyzer.saltbridge.export"
        ),
        "schema_version": "1.0",
        "module_version": __version__,
        "summary": (
            build_compact_salt_bridge_summary(
                result
            )
        ),
        "text_summary": (
            build_salt_bridge_text_summary(
                result
            )
        ),
        "pose_id": result.pose_id,
        "model_id": result.model_id,
    }

    if include_full_result:
        payload[
            "result"
        ] = salt_bridge_result_to_dict(
            result,
            include_interactions=True,
            include_groups=True,
            include_group_atoms=True,
            include_statistics=True,
            include_metadata=True,
            include_warnings=True,
            compact_interactions=False,
        )

    if include_interaction_table:
        payload[
            "interaction_rows"
        ] = salt_bridge_interactions_to_rows(
            result.interactions,
            include_invalid=True,
            sort_by_score=True,
        )

    if include_group_tables:
        payload[
            "cationic_group_rows"
        ] = salt_bridge_groups_to_rows(
            result.cationic_groups
        )

        payload[
            "anionic_group_rows"
        ] = salt_bridge_groups_to_rows(
            result.anionic_groups
        )

    if include_residue_summary:
        payload[
            "residue_summary_rows"
        ] = (
            build_residue_salt_bridge_summary_rows(
                result.interactions
            )
        )

    if include_statistics_table:
        payload[
            "statistics_rows"
        ] = salt_bridge_statistics_to_rows(
            result.statistics
        )

    return make_json_safe(
        payload
    )


def build_multipose_salt_bridge_export_payload(
    multipose_result: Mapping[str, Any],
    *,
    include_pose_results: bool = True,
    include_pose_interaction_rows: bool = True,
) -> Dict[str, Any]:
    """
    Build a complete multipose export payload.

    Parameters
    ----------
    multipose_result
        Complete multipose analysis result.
    include_pose_results
        Whether serialized pose results should be included.
    include_pose_interaction_rows
        Whether a unified interaction table should be included.

    Returns
    -------
    Dict[str, Any]
        Multipose export payload.
    """

    if not isinstance(
        multipose_result,
        Mapping,
    ):
        raise SaltBridgeSerializationError(
            "multipose_result must be a mapping."
        )

    results = [
        result
        for result in multipose_result.get(
            "results",
            [],
        )
        if isinstance(
            result,
            SaltBridgeResult,
        )
    ]

    payload: Dict[str, Any] = {
        "schema": (
            "dockanalyzer.saltbridge.multipose.export"
        ),
        "schema_version": "1.0",
        "module_version": __version__,
        "summary": make_json_safe(
            multipose_result.get(
                "compact_summary",
                {},
            )
        ),
        "text_summary": str(
            multipose_result.get(
                "text_summary",
                "",
            )
        ),
        "pose_summary_rows": (
            build_pose_salt_bridge_summary_rows(
                results
            )
        ),
        "persistence_rows": make_json_safe(
            multipose_result.get(
                "persistence",
                [],
            )
        ),
        "consensus_rows": make_json_safe(
            multipose_result.get(
                "consensus_interactions",
                [],
            )
        ),
        "statistics": make_json_safe(
            multipose_result.get(
                "statistics",
                {},
            )
        ),
        "metadata": make_json_safe(
            multipose_result.get(
                "metadata",
                {},
            )
        ),
        "warnings": make_json_safe(
            multipose_result.get(
                "warnings",
                [],
            )
        ),
    }

    if include_pose_results:
        payload[
            "results"
        ] = [
            salt_bridge_result_to_dict(
                result,
                include_interactions=True,
                include_groups=False,
                include_group_atoms=False,
                include_statistics=True,
                include_metadata=True,
                include_warnings=True,
                compact_interactions=True,
            )
            for result in results
        ]

    if include_pose_interaction_rows:
        interaction_rows: List[
            Dict[str, Any]
        ] = []

        for result in results:
            pose_rows = (
                salt_bridge_interactions_to_rows(
                    result.interactions,
                    include_invalid=True,
                    sort_by_score=True,
                )
            )

            for row in pose_rows:
                row.setdefault(
                    "pose_id",
                    result.pose_id,
                )

                row.setdefault(
                    "model_id",
                    result.model_id,
                )

                interaction_rows.append(
                    row
                )

        payload[
            "interaction_rows"
        ] = interaction_rows

    return make_json_safe(
        payload
    )


# =============================================================================
# 16.14. EXPORT FORMAT ROUTING
# =============================================================================


def normalize_salt_bridge_export_format(
    export_format: str,
) -> str:
    """
    Normalize an export format identifier.

    Parameters
    ----------
    export_format
        Export format name.

    Returns
    -------
    str
        Normalized format.

    Raises
    ------
    SaltBridgeSerializationError
        If the format is unsupported.
    """

    normalized_format = normalize_text(
        export_format,
        default="json",
        lowercase=True,
    )

    format_aliases = {
        "dictionary": "dict",
        "mapping": "dict",
        "json_string": "json",
        "table": "rows",
        "records": "rows",
        "interaction_rows": "rows",
        "summary": "compact_summary",
        "text": "text_summary",
    }

    normalized_format = (
        format_aliases.get(
            normalized_format,
            normalized_format,
        )
    )

    supported_formats = {
        "dict",
        "json",
        "rows",
        "compact_summary",
        "text_summary",
        "export_payload",
    }

    if normalized_format not in supported_formats:
        raise SaltBridgeSerializationError(
            "Unsupported salt-bridge export format: "
            f"{export_format!r}."
        )

    return normalized_format


def prepare_salt_bridge_export(
    result: SaltBridgeResult,
    export_format: str = "dict",
    **options: Any,
) -> Any:
    """
    Prepare a SaltBridgeResult in a requested export representation.

    This function does not write files.

    Parameters
    ----------
    result
        Salt-bridge result.
    export_format
        Output representation.
    **options
        Format-specific options.

    Returns
    -------
    Any
        Prepared export data.
    """

    normalized_format = (
        normalize_salt_bridge_export_format(
            export_format
        )
    )

    if normalized_format == "dict":
        return salt_bridge_result_to_dict(
            result,
            **options,
        )

    if normalized_format == "json":
        return serialize_salt_bridge_result(
            result,
            **options,
        )

    if normalized_format == "rows":
        return salt_bridge_interactions_to_rows(
            result.interactions,
            **options,
        )

    if normalized_format == "compact_summary":
        return build_compact_salt_bridge_summary(
            result
        )

    if normalized_format == "text_summary":
        return build_salt_bridge_text_summary(
            result
        )

    if normalized_format == "export_payload":
        return build_salt_bridge_export_payload(
            result,
            **options,
        )

    raise SaltBridgeSerializationError(
        "Internal export format routing failure."
    )


# =============================================================================
# 17. CHIMERAX COMPATIBILITY
# =============================================================================


# =============================================================================
# 17.1. CHIMERAX AVAILABILITY AND VALIDATION
# =============================================================================


def require_chimerax() -> None:
    """
    Ensure that ChimeraX integration is available.

    Raises
    ------
    ChimeraXUnavailableError
        If ChimeraX modules could not be imported.
    """

    if not HAS_CHIMERAX:
        raise ChimeraXUnavailableError(
            "ChimeraX integration is unavailable in the current environment."
        )


def is_chimerax_atomic_model(
    value: Any,
) -> bool:
    """
    Return whether an object appears to be a ChimeraX atomic model.

    Parameters
    ----------
    value
        Object to inspect.

    Returns
    -------
    bool
        Whether the object resembles a ChimeraX atomic model.
    """

    if value is None:
        return False

    if not HAS_CHIMERAX:
        return False

    candidate_attributes = (
        "atoms",
        "residues",
        "session",
        "id_string",
    )

    return all(
        hasattr(
            value,
            attribute_name,
        )
        for attribute_name in candidate_attributes
    )


def get_chimerax_session(
    value: Any,
    *,
    required: bool = True,
) -> Any:
    """
    Resolve a ChimeraX session from an object.

    Parameters
    ----------
    value
        Session-like object or object containing a session.
    required
        Whether failure should raise an exception.

    Returns
    -------
    Any
        Resolved ChimeraX session or ``None``.

    Raises
    ------
    ChimeraXSaltBridgeError
        If no session can be resolved and ``required`` is true.
    """

    if value is None:
        if required:
            raise ChimeraXSaltBridgeError(
                "A ChimeraX session is required."
            )

        return None

    if hasattr(
        value,
        "models",
    ) and hasattr(
        value,
        "logger",
    ):
        return value

    session = get_value(
        value,
        "session",
        None,
    )

    if session is not None:
        return session

    if required:
        raise ChimeraXSaltBridgeError(
            "Could not resolve a ChimeraX session."
        )

    return None


# =============================================================================
# 17.2. CHIMERAX MODEL SPECIFICATIONS
# =============================================================================


def get_chimerax_model_spec(
    model: Any,
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """
    Build a ChimeraX model specification.

    Parameters
    ----------
    model
        ChimeraX model-like object.
    fallback
        Value returned when no model identifier is available.

    Returns
    -------
    Optional[str]
        ChimeraX model specification.
    """

    if model is None:
        return fallback

    id_string = get_value(
        model,
        "id_string",
        None,
    )

    if id_string:
        normalized_id = str(
            id_string
        ).strip()

        if normalized_id.startswith(
            "#"
        ):
            return normalized_id

        return f"#{normalized_id}"

    model_id = get_value(
        model,
        "id",
        None,
    )

    if model_id is not None:
        if isinstance(
            model_id,
            (
                tuple,
                list,
            ),
        ):
            identifier = ".".join(
                str(item)
                for item in model_id
            )

        else:
            identifier = str(
                model_id
            )

        if identifier:
            return f"#{identifier}"

    return fallback


def get_atom_chimerax_model(
    atom: Any,
) -> Any:
    """
    Resolve the ChimeraX model associated with an atom.

    Parameters
    ----------
    atom
        Atom-like object.

    Returns
    -------
    Any
        ChimeraX model-like object or ``None``.
    """

    if atom is None:
        return None

    structure = get_value(
        atom,
        "structure",
        None,
    )

    if structure is not None:
        return structure

    residue = get_atom_residue(
        atom
    )

    if residue is not None:
        structure = get_value(
            residue,
            "structure",
            None,
        )

        if structure is not None:
            return structure

    return None


# =============================================================================
# 17.3. ATOM SPECIFICATIONS
# =============================================================================


def escape_chimerax_spec_text(
    value: Any,
) -> str:
    """
    Escape text used in ChimeraX atom specifications.

    Parameters
    ----------
    value
        Text-like value.

    Returns
    -------
    str
        Sanitized specification component.
    """

    text = str(
        value
    ).strip()

    return (
        text.replace(
            "\\",
            "\\\\",
        )
        .replace(
            '"',
            '\\"',
        )
    )


def atom_to_chimerax_spec(
    atom: Any,
    *,
    include_model: bool = True,
    include_chain: bool = True,
    include_residue: bool = True,
    include_atom_name: bool = True,
) -> Optional[str]:
    """
    Build a ChimeraX specification for one atom.

    Parameters
    ----------
    atom
        Atom-like object.
    include_model
        Whether the model identifier should be included.
    include_chain
        Whether the chain identifier should be included.
    include_residue
        Whether the residue identifier should be included.
    include_atom_name
        Whether the atom name should be included.

    Returns
    -------
    Optional[str]
        ChimeraX atom specification or ``None``.
    """

    if atom is None:
        return None

    specification_parts: List[str] = []

    if include_model:
        model = get_atom_chimerax_model(
            atom
        )

        model_spec = get_chimerax_model_spec(
            model
        )

        if model_spec:
            specification_parts.append(
                model_spec
            )

    residue = get_atom_residue(
        atom
    )

    if include_chain and residue is not None:
        chain_id = get_chain_id(
            residue
        )

        if chain_id:
            specification_parts.append(
                f"/{escape_chimerax_spec_text(chain_id)}"
            )

    if include_residue and residue is not None:
        residue_number = get_residue_number(
            residue
        )

        if residue_number is not None:
            specification_parts.append(
                f":{escape_chimerax_spec_text(residue_number)}"
            )

    if include_atom_name:
        atom_name = get_atom_name(
            atom
        )

        if atom_name:
            specification_parts.append(
                f"@{escape_chimerax_spec_text(atom_name)}"
            )

    if not specification_parts:
        serial = get_atom_serial(
            atom
        )

        if serial is not None:
            return f"@serial_number={serial}"

        return None

    return "".join(
        specification_parts
    )


def atoms_to_chimerax_spec(
    atoms: Iterable[Any],
    *,
    operator: str = " ",
) -> Optional[str]:
    """
    Build a combined ChimeraX specification for multiple atoms.

    Parameters
    ----------
    atoms
        Atom-like objects.
    operator
        Specification separator.

    Returns
    -------
    Optional[str]
        Combined ChimeraX specification.
    """

    specifications = unique_preserve_order(
        specification
        for specification in (
            atom_to_chimerax_spec(
                atom
            )
            for atom in atoms
        )
        if specification
    )

    if not specifications:
        return None

    return operator.join(
        specifications
    )


# =============================================================================
# 17.4. RESIDUE SPECIFICATIONS
# =============================================================================


def residue_to_chimerax_spec(
    residue: Any,
    *,
    include_model: bool = True,
    include_chain: bool = True,
) -> Optional[str]:
    """
    Build a ChimeraX specification for one residue.

    Parameters
    ----------
    residue
        Residue-like object.
    include_model
        Whether the model identifier should be included.
    include_chain
        Whether the chain identifier should be included.

    Returns
    -------
    Optional[str]
        ChimeraX residue specification.
    """

    if residue is None:
        return None

    specification_parts: List[str] = []

    if include_model:
        model = get_value(
            residue,
            "structure",
            None,
        )

        model_spec = get_chimerax_model_spec(
            model
        )

        if model_spec:
            specification_parts.append(
                model_spec
            )

    if include_chain:
        chain_id = get_chain_id(
            residue
        )

        if chain_id:
            specification_parts.append(
                f"/{escape_chimerax_spec_text(chain_id)}"
            )

    residue_number = get_residue_number(
        residue
    )

    if residue_number is not None:
        specification_parts.append(
            f":{escape_chimerax_spec_text(residue_number)}"
        )

    if not specification_parts:
        return None

    return "".join(
        specification_parts
    )


def residues_to_chimerax_spec(
    residues: Iterable[Any],
    *,
    operator: str = " ",
) -> Optional[str]:
    """
    Build a combined ChimeraX specification for residues.

    Parameters
    ----------
    residues
        Residue-like objects.
    operator
        Specification separator.

    Returns
    -------
    Optional[str]
        Combined ChimeraX specification.
    """

    specifications = unique_preserve_order(
        specification
        for specification in (
            residue_to_chimerax_spec(
                residue
            )
            for residue in residues
        )
        if specification
    )

    if not specifications:
        return None

    return operator.join(
        specifications
    )


# =============================================================================
# 17.5. CHARGED GROUP SPECIFICATIONS
# =============================================================================


def charged_group_to_chimerax_spec(
    group: ChargedGroup,
    *,
    representative_only: bool = False,
) -> Optional[str]:
    """
    Build a ChimeraX specification for a charged group.

    Parameters
    ----------
    group
        Charged group.
    representative_only
        Whether only the representative atom should be selected.

    Returns
    -------
    Optional[str]
        ChimeraX atom specification.
    """

    if not isinstance(
        group,
        ChargedGroup,
    ):
        raise ChimeraXSaltBridgeError(
            "group must be a ChargedGroup instance."
        )

    if (
        representative_only
        and group.representative_atom is not None
    ):
        return atom_to_chimerax_spec(
            group.representative_atom
        )

    group_atoms = [
        charged_atom.atom
        for charged_atom in group.atoms
        if charged_atom.atom is not None
    ]

    if group_atoms:
        return atoms_to_chimerax_spec(
            group_atoms
        )

    if group.representative_atom is not None:
        return atom_to_chimerax_spec(
            group.representative_atom
        )

    return residue_to_chimerax_spec(
        group.residue
    )


def salt_bridge_interaction_to_chimerax_spec(
    interaction: SaltBridgeInteraction,
    *,
    residues_only: bool = False,
    representative_atoms_only: bool = False,
) -> Optional[str]:
    """
    Build a ChimeraX specification for one salt bridge.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    residues_only
        Whether entire residues should be selected.
    representative_atoms_only
        Whether only representative group atoms should be selected.

    Returns
    -------
    Optional[str]
        Combined ChimeraX specification.
    """

    if not isinstance(
        interaction,
        SaltBridgeInteraction,
    ):
        raise ChimeraXSaltBridgeError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    if residues_only:
        return residues_to_chimerax_spec(
            (
                interaction.cation.residue,
                interaction.anion.residue,
            )
        )

    cation_spec = charged_group_to_chimerax_spec(
        interaction.cation,
        representative_only=(
            representative_atoms_only
        ),
    )

    anion_spec = charged_group_to_chimerax_spec(
        interaction.anion,
        representative_only=(
            representative_atoms_only
        ),
    )

    specifications = [
        specification
        for specification in (
            cation_spec,
            anion_spec,
        )
        if specification
    ]

    if not specifications:
        return None

    return " ".join(
        specifications
    )


# =============================================================================
# 17.6. RESULT SPECIFICATIONS
# =============================================================================


def salt_bridge_result_to_chimerax_spec(
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    residues_only: bool = False,
    representative_atoms_only: bool = False,
) -> Optional[str]:
    """
    Build a ChimeraX selection specification for a complete result.

    Parameters
    ----------
    result
        Salt-bridge result.
    valid_only
        Whether rejected interactions should be excluded.
    residues_only
        Whether complete residues should be selected.
    representative_atoms_only
        Whether only representative atoms should be selected.

    Returns
    -------
    Optional[str]
        Combined ChimeraX specification.
    """

    if not isinstance(
        result,
        SaltBridgeResult,
    ):
        raise ChimeraXSaltBridgeError(
            "result must be a SaltBridgeResult instance."
        )

    interaction_specs: List[str] = []

    for interaction in result.interactions:
        if (
            valid_only
            and not interaction.geometry.valid
        ):
            continue

        interaction_spec = (
            salt_bridge_interaction_to_chimerax_spec(
                interaction,
                residues_only=residues_only,
                representative_atoms_only=(
                    representative_atoms_only
                ),
            )
        )

        if interaction_spec:
            interaction_specs.append(
                interaction_spec
            )

    unique_specs = unique_preserve_order(
        interaction_specs
    )

    if not unique_specs:
        return None

    return " ".join(
        unique_specs
    )


# =============================================================================
# 17.7. SELECTION COMMANDS
# =============================================================================


def build_select_salt_bridge_command(
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    residues_only: bool = False,
    representative_atoms_only: bool = False,
    clear_existing: bool = True,
) -> Optional[str]:
    """
    Build a ChimeraX selection command for salt bridges.

    Parameters
    ----------
    result
        Salt-bridge result.
    valid_only
        Whether invalid interactions should be excluded.
    residues_only
        Whether complete residues should be selected.
    representative_atoms_only
        Whether only representative atoms should be selected.
    clear_existing
        Whether the existing selection should be cleared.

    Returns
    -------
    Optional[str]
        ChimeraX command string.
    """

    atom_spec = salt_bridge_result_to_chimerax_spec(
        result,
        valid_only=valid_only,
        residues_only=residues_only,
        representative_atoms_only=(
            representative_atoms_only
        ),
    )

    if not atom_spec:
        return None

    select_command = f"select {atom_spec}"

    if clear_existing:
        return (
            "select clear; "
            + select_command
        )

    return select_command


def build_select_interaction_command(
    interaction: SaltBridgeInteraction,
    *,
    residues_only: bool = False,
    representative_atoms_only: bool = False,
    clear_existing: bool = True,
) -> Optional[str]:
    """
    Build a selection command for one salt bridge.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    residues_only
        Whether complete residues should be selected.
    representative_atoms_only
        Whether only representative atoms should be selected.
    clear_existing
        Whether the existing selection should be cleared.

    Returns
    -------
    Optional[str]
        ChimeraX command.
    """

    atom_spec = (
        salt_bridge_interaction_to_chimerax_spec(
            interaction,
            residues_only=residues_only,
            representative_atoms_only=(
                representative_atoms_only
            ),
        )
    )

    if not atom_spec:
        return None

    command = f"select {atom_spec}"

    if clear_existing:
        return f"select clear; {command}"

    return command


# =============================================================================
# 17.8. DISPLAY COMMANDS
# =============================================================================


def normalize_chimerax_color(
    color: Optional[str],
    *,
    default: str,
) -> str:
    """
    Normalize a ChimeraX color name.

    Parameters
    ----------
    color
        Color name.
    default
        Fallback color.

    Returns
    -------
    str
        Normalized color name.
    """

    normalized_color = normalize_text(
        color,
        default=default,
        lowercase=True,
    )

    return (
        normalized_color
        or default
    )


def get_salt_bridge_strength_color(
    interaction: SaltBridgeInteraction,
) -> str:
    """
    Return a default ChimeraX color based on interaction strength.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    str
        ChimeraX color name.
    """

    strength = normalize_text(
        interaction.strength,
        default=STRENGTH_REJECTED,
        lowercase=True,
    )

    color_mapping = {
        STRENGTH_STRONG: "magenta",
        STRENGTH_MODERATE: "purple",
        STRENGTH_WEAK: "orchid",
        STRENGTH_REJECTED: "gray",
    }

    return color_mapping.get(
        strength,
        "magenta",
    )


def build_show_salt_bridge_command(
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    residues_only: bool = False,
    display_style: str = "stick",
    color: Optional[str] = None,
) -> Optional[str]:
    """
    Build ChimeraX commands to show salt-bridge participants.

    Parameters
    ----------
    result
        Salt-bridge result.
    valid_only
        Whether invalid interactions should be excluded.
    residues_only
        Whether complete residues should be displayed.
    display_style
        ChimeraX display style.
    color
        Optional display color.

    Returns
    -------
    Optional[str]
        ChimeraX command string.
    """

    atom_spec = salt_bridge_result_to_chimerax_spec(
        result,
        valid_only=valid_only,
        residues_only=residues_only,
    )

    if not atom_spec:
        return None

    normalized_style = normalize_text(
        display_style,
        default="stick",
        lowercase=True,
    )

    command_parts = [
        f"show {atom_spec}",
        f"style {atom_spec} {normalized_style}",
    ]

    if color:
        normalized_color = normalize_chimerax_color(
            color,
            default="magenta",
        )

        command_parts.append(
            f"color {atom_spec} {normalized_color}"
        )

    return "; ".join(
        command_parts
    )


# =============================================================================
# 17.9. PSEUDOBOND ENDPOINT RESOLUTION
# =============================================================================


def resolve_salt_bridge_pseudobond_atoms(
    interaction: SaltBridgeInteraction,
) -> Tuple[Any, Any]:
    """
    Resolve atoms used as salt-bridge pseudobond endpoints.

    Priority:

    1. closest atomic contact;
    2. representative atoms;
    3. first available atom in each charged group.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Tuple[Any, Any]
        Positive and negative endpoint atoms.

    Raises
    ------
    ChimeraXSaltBridgeError
        If suitable atoms cannot be resolved.
    """

    if not isinstance(
        interaction,
        SaltBridgeInteraction,
    ):
        raise ChimeraXSaltBridgeError(
            "interaction must be a SaltBridgeInteraction instance."
        )

    positive_atom = (
        interaction.geometry.closest_positive_atom
    )

    negative_atom = (
        interaction.geometry.closest_negative_atom
    )

    if positive_atom is None:
        positive_atom = (
            interaction.cation.representative_atom
        )

    if negative_atom is None:
        negative_atom = (
            interaction.anion.representative_atom
        )

    if (
        positive_atom is None
        and interaction.cation.atoms
    ):
        positive_atom = (
            interaction.cation.atoms[0].atom
        )

    if (
        negative_atom is None
        and interaction.anion.atoms
    ):
        negative_atom = (
            interaction.anion.atoms[0].atom
        )

    if (
        positive_atom is None
        or negative_atom is None
    ):
        raise ChimeraXSaltBridgeError(
            "Could not resolve pseudobond endpoint atoms."
        )

    return (
        positive_atom,
        negative_atom,
    )


def build_salt_bridge_pseudobond_specs(
    interaction: SaltBridgeInteraction,
) -> Tuple[str, str]:
    """
    Build ChimeraX atom specifications for pseudobond endpoints.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Tuple[str, str]
        Positive and negative atom specifications.
    """

    positive_atom, negative_atom = (
        resolve_salt_bridge_pseudobond_atoms(
            interaction
        )
    )

    positive_spec = atom_to_chimerax_spec(
        positive_atom
    )

    negative_spec = atom_to_chimerax_spec(
        negative_atom
    )

    if (
        not positive_spec
        or not negative_spec
    ):
        raise ChimeraXSaltBridgeError(
            "Could not build pseudobond atom specifications."
        )

    return (
        positive_spec,
        negative_spec,
    )


# =============================================================================
# 17.10. PSEUDOBOND GROUP NAMING
# =============================================================================


def sanitize_chimerax_group_name(
    value: Any,
    *,
    default: str = "DockAnalyzer salt bridges",
) -> str:
    """
    Sanitize a ChimeraX pseudobond group name.

    Parameters
    ----------
    value
        Group name.
    default
        Fallback name.

    Returns
    -------
    str
        Sanitized group name.
    """

    normalized_name = str(
        value or default
    ).strip()

    if not normalized_name:
        normalized_name = default

    return normalized_name.replace(
        '"',
        "'",
    )


def make_salt_bridge_pseudobond_group_name(
    interaction: Optional[
        SaltBridgeInteraction
    ] = None,
    *,
    base_name: str = "DockAnalyzer salt bridges",
    separate_by_strength: bool = False,
    separate_by_pose: bool = False,
) -> str:
    """
    Build a pseudobond group name.

    Parameters
    ----------
    interaction
        Optional salt-bridge interaction.
    base_name
        Base group name.
    separate_by_strength
        Whether strength should be included.
    separate_by_pose
        Whether pose identifier should be included.

    Returns
    -------
    str
        Pseudobond group name.
    """

    name_parts = [
        sanitize_chimerax_group_name(
            base_name
        )
    ]

    if interaction is not None:
        if separate_by_strength:
            strength = normalize_text(
                interaction.strength,
                default="unclassified",
                lowercase=True,
            )

            name_parts.append(
                strength
            )

        if (
            separate_by_pose
            and interaction.pose_id is not None
        ):
            name_parts.append(
                f"pose {interaction.pose_id}"
            )

    return " - ".join(
        name_parts
    )


# =============================================================================
# 17.11. PSEUDOBOND COMMAND GENERATION
# =============================================================================


def build_create_salt_bridge_pseudobond_command(
    interaction: SaltBridgeInteraction,
    *,
    group_name: Optional[str] = None,
    color: Optional[str] = None,
    radius: float = 0.15,
    dashes: int = 6,
    separate_by_strength: bool = False,
    separate_by_pose: bool = False,
) -> str:
    """
    Build a ChimeraX command to create one salt-bridge pseudobond.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.
    group_name
        Optional pseudobond group name.
    color
        Optional pseudobond color.
    radius
        Pseudobond radius.
    dashes
        Number of pseudobond dashes.
    separate_by_strength
        Whether group names should include strength.
    separate_by_pose
        Whether group names should include pose identifier.

    Returns
    -------
    str
        ChimeraX command string.
    """

    positive_spec, negative_spec = (
        build_salt_bridge_pseudobond_specs(
            interaction
        )
    )

    resolved_group_name = (
        make_salt_bridge_pseudobond_group_name(
            interaction,
            base_name=(
                group_name
                or "DockAnalyzer salt bridges"
            ),
            separate_by_strength=(
                separate_by_strength
            ),
            separate_by_pose=(
                separate_by_pose
            ),
        )
    )

    resolved_color = (
        normalize_chimerax_color(
            color,
            default=(
                get_salt_bridge_strength_color(
                    interaction
                )
            ),
        )
    )

    normalized_radius = safe_float(
        radius,
        default=0.15,
    )

    normalized_dashes = safe_int(
        dashes,
        default=6,
    )

    if (
        normalized_radius is None
        or normalized_radius <= 0.0
    ):
        raise ChimeraXSaltBridgeError(
            "Pseudobond radius must be positive."
        )

    if (
        normalized_dashes is None
        or normalized_dashes < 0
    ):
        raise ChimeraXSaltBridgeError(
            "Pseudobond dash count cannot be negative."
        )

    escaped_group_name = (
        resolved_group_name.replace(
            '"',
            '\\"',
        )
    )

    return (
        f'pbond {positive_spec} {negative_spec} '
        f'color {resolved_color} '
        f'radius {normalized_radius:.3f} '
        f'dashes {normalized_dashes} '
        f'name "{escaped_group_name}"'
    )


def build_create_salt_bridge_pseudobonds_commands(
    interactions: Iterable[
        SaltBridgeInteraction
    ],
    *,
    valid_only: bool = True,
    group_name: str = "DockAnalyzer salt bridges",
    color_by_strength: bool = True,
    color: Optional[str] = None,
    radius: float = 0.15,
    dashes: int = 6,
    separate_by_strength: bool = False,
    separate_by_pose: bool = False,
) -> List[str]:
    """
    Build ChimeraX pseudobond creation commands.

    Parameters
    ----------
    interactions
        Salt-bridge interactions.
    valid_only
        Whether invalid interactions should be excluded.
    group_name
        Base pseudobond group name.
    color_by_strength
        Whether interaction strength should determine color.
    color
        Optional fixed color.
    radius
        Pseudobond radius.
    dashes
        Number of dashes.
    separate_by_strength
        Whether separate strength groups should be created.
    separate_by_pose
        Whether separate pose groups should be created.

    Returns
    -------
    List[str]
        ChimeraX commands.
    """

    commands: List[str] = []

    for interaction in interactions:
        if not isinstance(
            interaction,
            SaltBridgeInteraction,
        ):
            raise ChimeraXSaltBridgeError(
                "All values must be SaltBridgeInteraction instances."
            )

        if (
            valid_only
            and not interaction.geometry.valid
        ):
            continue

        interaction_color = (
            None
            if color_by_strength
            else color
        )

        if (
            color_by_strength
            and color is not None
        ):
            interaction_color = color

        command = (
            build_create_salt_bridge_pseudobond_command(
                interaction,
                group_name=group_name,
                color=interaction_color,
                radius=radius,
                dashes=dashes,
                separate_by_strength=(
                    separate_by_strength
                ),
                separate_by_pose=(
                    separate_by_pose
                ),
            )
        )

        commands.append(
            command
        )

    return commands


def build_create_result_pseudobonds_commands(
    result: SaltBridgeResult,
    **options: Any,
) -> List[str]:
    """
    Build pseudobond commands for a complete SaltBridgeResult.

    Parameters
    ----------
    result
        Salt-bridge result.
    **options
        Options forwarded to
        ``build_create_salt_bridge_pseudobonds_commands``.

    Returns
    -------
    List[str]
        ChimeraX commands.
    """

    if not isinstance(
        result,
        SaltBridgeResult,
    ):
        raise ChimeraXSaltBridgeError(
            "result must be a SaltBridgeResult instance."
        )

    return build_create_salt_bridge_pseudobonds_commands(
        result.interactions,
        **options,
    )


# =============================================================================
# 17.12. PSEUDOBOND DELETION COMMANDS
# =============================================================================


def build_delete_salt_bridge_pseudobonds_command(
    *,
    group_name: str = "DockAnalyzer salt bridges",
) -> str:
    """
    Build a command to delete a salt-bridge pseudobond group.

    Parameters
    ----------
    group_name
        Pseudobond group name.

    Returns
    -------
    str
        ChimeraX command.
    """

    normalized_group_name = (
        sanitize_chimerax_group_name(
            group_name
        )
    )

    escaped_group_name = (
        normalized_group_name.replace(
            '"',
            '\\"',
        )
    )

    return (
        f'pbond delete name "{escaped_group_name}"'
    )


def build_hide_salt_bridge_pseudobonds_command(
    *,
    group_name: str = "DockAnalyzer salt bridges",
) -> str:
    """
    Build a command to hide a salt-bridge pseudobond group.

    Parameters
    ----------
    group_name
        Pseudobond group name.

    Returns
    -------
    str
        ChimeraX command.
    """

    normalized_group_name = (
        sanitize_chimerax_group_name(
            group_name
        )
    )

    escaped_group_name = (
        normalized_group_name.replace(
            '"',
            '\\"',
        )
    )

    return (
        f'hide pseudobonds name "{escaped_group_name}"'
    )


def build_show_salt_bridge_pseudobonds_command(
    *,
    group_name: str = "DockAnalyzer salt bridges",
) -> str:
    """
    Build a command to show a salt-bridge pseudobond group.

    Parameters
    ----------
    group_name
        Pseudobond group name.

    Returns
    -------
    str
        ChimeraX command.
    """

    normalized_group_name = (
        sanitize_chimerax_group_name(
            group_name
        )
    )

    escaped_group_name = (
        normalized_group_name.replace(
            '"',
            '\\"',
        )
    )

    return (
        f'show pseudobonds name "{escaped_group_name}"'
    )


# =============================================================================
# 17.13. CHIMERAX COMMAND EXECUTION
# =============================================================================


def run_chimerax_command(
    session: Any,
    command: str,
    *,
    log: bool = False,
) -> Any:
    """
    Execute one ChimeraX command.

    Parameters
    ----------
    session
        ChimeraX session.
    command
        Command string.
    log
        Whether the command should be logged by ChimeraX.

    Returns
    -------
    Any
        Command result.

    Raises
    ------
    ChimeraXUnavailableError
        If ChimeraX is unavailable.
    ChimeraXSaltBridgeError
        If command execution fails.
    """

    require_chimerax()

    resolved_session = get_chimerax_session(
        session,
        required=True,
    )

    normalized_command = str(
        command
    ).strip()

    if not normalized_command:
        raise ChimeraXSaltBridgeError(
            "A non-empty ChimeraX command is required."
        )

    try:
        return chimerax_run(
            resolved_session,
            normalized_command,
            log=log,
        )

    except Exception as error:
        raise ChimeraXSaltBridgeError(
            "ChimeraX command execution failed: "
            f"{normalized_command}"
        ) from error


def run_chimerax_commands(
    session: Any,
    commands: Iterable[str],
    *,
    log: bool = False,
    continue_on_error: bool = False,
    warnings: Optional[List[str]] = None,
) -> List[Any]:
    """
    Execute multiple ChimeraX commands.

    Parameters
    ----------
    session
        ChimeraX session.
    commands
        Command strings.
    log
        Whether commands should be logged.
    continue_on_error
        Whether execution should continue after failures.
    warnings
        Optional warning collector.

    Returns
    -------
    List[Any]
        Command results.
    """

    command_results: List[Any] = []

    for command_index, command in enumerate(
        commands,
        start=1,
    ):
        try:
            command_result = run_chimerax_command(
                session,
                command,
                log=log,
            )

            command_results.append(
                command_result
            )

        except ChimeraXSaltBridgeError as error:
            message = (
                "ChimeraX salt-bridge command failed "
                f"at index {command_index}: {error}"
            )

            if warnings is not None:
                warnings.append(
                    message
                )

            if not continue_on_error:
                raise

    return command_results


# =============================================================================
# 17.14. DIRECT SELECTION AND VISUALIZATION
# =============================================================================


def select_salt_bridges_in_chimerax(
    session: Any,
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    residues_only: bool = False,
    representative_atoms_only: bool = False,
    clear_existing: bool = True,
    log: bool = False,
) -> Any:
    """
    Select salt-bridge participants in ChimeraX.

    Parameters
    ----------
    session
        ChimeraX session.
    result
        Salt-bridge result.
    valid_only
        Whether rejected interactions should be excluded.
    residues_only
        Whether complete residues should be selected.
    representative_atoms_only
        Whether only representative atoms should be selected.
    clear_existing
        Whether the previous selection should be cleared.
    log
        Whether the command should be logged.

    Returns
    -------
    Any
        ChimeraX command result.
    """

    command = build_select_salt_bridge_command(
        result,
        valid_only=valid_only,
        residues_only=residues_only,
        representative_atoms_only=(
            representative_atoms_only
        ),
        clear_existing=clear_existing,
    )

    if command is None:
        return None

    return run_chimerax_command(
        session,
        command,
        log=log,
    )


def create_salt_bridge_pseudobonds_in_chimerax(
    session: Any,
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    group_name: str = "DockAnalyzer salt bridges",
    color_by_strength: bool = True,
    color: Optional[str] = None,
    radius: float = 0.15,
    dashes: int = 6,
    separate_by_strength: bool = False,
    separate_by_pose: bool = False,
    clear_existing: bool = False,
    continue_on_error: bool = False,
    warnings: Optional[List[str]] = None,
    log: bool = False,
) -> List[Any]:
    """
    Create salt-bridge pseudobonds in ChimeraX.

    Parameters
    ----------
    session
        ChimeraX session.
    result
        Salt-bridge result.
    valid_only
        Whether rejected interactions should be excluded.
    group_name
        Base pseudobond group name.
    color_by_strength
        Whether pseudobonds should be colored by strength.
    color
        Optional fixed color.
    radius
        Pseudobond radius.
    dashes
        Number of pseudobond dashes.
    separate_by_strength
        Whether separate strength groups should be created.
    separate_by_pose
        Whether separate pose groups should be created.
    clear_existing
        Whether the existing pseudobond group should be deleted first.
    continue_on_error
        Whether execution should continue after command failures.
    warnings
        Optional warning collector.
    log
        Whether commands should be logged.

    Returns
    -------
    List[Any]
        ChimeraX command results.
    """

    require_chimerax()

    commands: List[str] = []

    if clear_existing:
        commands.append(
            build_delete_salt_bridge_pseudobonds_command(
                group_name=group_name
            )
        )

    commands.extend(
        build_create_result_pseudobonds_commands(
            result,
            valid_only=valid_only,
            group_name=group_name,
            color_by_strength=(
                color_by_strength
            ),
            color=color,
            radius=radius,
            dashes=dashes,
            separate_by_strength=(
                separate_by_strength
            ),
            separate_by_pose=(
                separate_by_pose
            ),
        )
    )

    return run_chimerax_commands(
        session,
        commands,
        log=log,
        continue_on_error=(
            continue_on_error
        ),
        warnings=warnings,
    )


# =============================================================================
# 17.15. COMPLETE VISUALIZATION COMMAND SET
# =============================================================================


def build_salt_bridge_visualization_commands(
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
    group_name: str = "DockAnalyzer salt bridges",
    display_residues: bool = True,
    display_style: str = "stick",
    color_participants: bool = False,
    participant_color: str = "magenta",
    color_by_strength: bool = True,
    pseudobond_color: Optional[str] = None,
    pseudobond_radius: float = 0.15,
    pseudobond_dashes: int = 6,
    separate_by_strength: bool = False,
    separate_by_pose: bool = False,
    clear_existing_pseudobonds: bool = True,
    select_participants: bool = True,
) -> List[str]:
    """
    Build a complete ChimeraX visualization command set.

    Parameters
    ----------
    result
        Salt-bridge result.
    valid_only
        Whether invalid interactions should be excluded.
    group_name
        Pseudobond group name.
    display_residues
        Whether entire interacting residues should be displayed.
    display_style
        Participant display style.
    color_participants
        Whether interacting atoms or residues should be colored.
    participant_color
        Participant color.
    color_by_strength
        Whether pseudobonds should be colored by strength.
    pseudobond_color
        Optional fixed pseudobond color.
    pseudobond_radius
        Pseudobond radius.
    pseudobond_dashes
        Pseudobond dash count.
    separate_by_strength
        Whether pseudobonds should be grouped by strength.
    separate_by_pose
        Whether pseudobonds should be grouped by pose.
    clear_existing_pseudobonds
        Whether existing pseudobonds should be deleted first.
    select_participants
        Whether participants should be selected.

    Returns
    -------
    List[str]
        ChimeraX command strings.
    """

    commands: List[str] = []

    if clear_existing_pseudobonds:
        commands.append(
            build_delete_salt_bridge_pseudobonds_command(
                group_name=group_name
            )
        )

    display_command = (
        build_show_salt_bridge_command(
            result,
            valid_only=valid_only,
            residues_only=display_residues,
            display_style=display_style,
            color=(
                participant_color
                if color_participants
                else None
            ),
        )
    )

    if display_command:
        commands.append(
            display_command
        )

    if select_participants:
        selection_command = (
            build_select_salt_bridge_command(
                result,
                valid_only=valid_only,
                residues_only=display_residues,
                representative_atoms_only=False,
                clear_existing=True,
            )
        )

        if selection_command:
            commands.append(
                selection_command
            )

    commands.extend(
        build_create_result_pseudobonds_commands(
            result,
            valid_only=valid_only,
            group_name=group_name,
            color_by_strength=(
                color_by_strength
            ),
            color=pseudobond_color,
            radius=pseudobond_radius,
            dashes=pseudobond_dashes,
            separate_by_strength=(
                separate_by_strength
            ),
            separate_by_pose=(
                separate_by_pose
            ),
        )
    )

    return commands


def visualize_salt_bridges_in_chimerax(
    session: Any,
    result: SaltBridgeResult,
    *,
    continue_on_error: bool = False,
    warnings: Optional[List[str]] = None,
    log: bool = False,
    **visualization_options: Any,
) -> List[Any]:
    """
    Visualize salt bridges in ChimeraX.

    Parameters
    ----------
    session
        ChimeraX session.
    result
        Salt-bridge result.
    continue_on_error
        Whether command execution should continue after failures.
    warnings
        Optional warning collector.
    log
        Whether commands should be logged.
    **visualization_options
        Options forwarded to
        ``build_salt_bridge_visualization_commands``.

    Returns
    -------
    List[Any]
        ChimeraX command results.
    """

    commands = (
        build_salt_bridge_visualization_commands(
            result,
            **visualization_options,
        )
    )

    return run_chimerax_commands(
        session,
        commands,
        log=log,
        continue_on_error=(
            continue_on_error
        ),
        warnings=warnings,
    )


# =============================================================================
# 17.16. MULTIPOSE CHIMERAX VISUALIZATION
# =============================================================================


def build_multipose_salt_bridge_visualization_commands(
    results: Iterable[SaltBridgeResult],
    *,
    valid_only: bool = True,
    base_group_name: str = "DockAnalyzer salt bridges",
    separate_by_pose: bool = True,
    separate_by_strength: bool = False,
    color_by_strength: bool = True,
    pseudobond_radius: float = 0.15,
    pseudobond_dashes: int = 6,
) -> List[str]:
    """
    Build pseudobond commands for multiple pose results.

    Parameters
    ----------
    results
        Pose-level salt-bridge results.
    valid_only
        Whether invalid interactions should be excluded.
    base_group_name
        Base pseudobond group name.
    separate_by_pose
        Whether each pose should use a separate group.
    separate_by_strength
        Whether strength should also define groups.
    color_by_strength
        Whether pseudobond colors should reflect strength.
    pseudobond_radius
        Pseudobond radius.
    pseudobond_dashes
        Pseudobond dash count.

    Returns
    -------
    List[str]
        ChimeraX commands.
    """

    commands: List[str] = []

    for result in results:
        if not isinstance(
            result,
            SaltBridgeResult,
        ):
            raise ChimeraXSaltBridgeError(
                "All values must be SaltBridgeResult instances."
            )

        commands.extend(
            build_create_result_pseudobonds_commands(
                result,
                valid_only=valid_only,
                group_name=base_group_name,
                color_by_strength=(
                    color_by_strength
                ),
                radius=pseudobond_radius,
                dashes=pseudobond_dashes,
                separate_by_strength=(
                    separate_by_strength
                ),
                separate_by_pose=(
                    separate_by_pose
                ),
            )
        )

    return commands


def visualize_multipose_salt_bridges_in_chimerax(
    session: Any,
    results: Iterable[SaltBridgeResult],
    *,
    continue_on_error: bool = True,
    warnings: Optional[List[str]] = None,
    log: bool = False,
    **options: Any,
) -> List[Any]:
    """
    Visualize salt bridges from multiple poses in ChimeraX.

    Parameters
    ----------
    session
        ChimeraX session.
    results
        Pose-level results.
    continue_on_error
        Whether command execution should continue after failures.
    warnings
        Optional warning collector.
    log
        Whether commands should be logged.
    **options
        Visualization options.

    Returns
    -------
    List[Any]
        ChimeraX command results.
    """

    commands = (
        build_multipose_salt_bridge_visualization_commands(
            results,
            **options,
        )
    )

    return run_chimerax_commands(
        session,
        commands,
        log=log,
        continue_on_error=(
            continue_on_error
        ),
        warnings=warnings,
    )


# =============================================================================
# 17.17. CHIMERAX EXPORT RECORDS
# =============================================================================


def build_chimerax_salt_bridge_record(
    interaction: SaltBridgeInteraction,
) -> Dict[str, Any]:
    """
    Build a ChimeraX-oriented interaction record.

    Parameters
    ----------
    interaction
        Salt-bridge interaction.

    Returns
    -------
    Dict[str, Any]
        ChimeraX-oriented record.
    """

    positive_spec, negative_spec = (
        build_salt_bridge_pseudobond_specs(
            interaction
        )
    )

    return {
        "interaction_id": (
            interaction.interaction_id
        ),
        "pose_id": interaction.pose_id,
        "model_id": interaction.model_id,
        "strength": interaction.strength,
        "score": sanitize_json_number(
            interaction.score
        ),
        "positive_atom_spec": (
            positive_spec
        ),
        "negative_atom_spec": (
            negative_spec
        ),
        "cation_group_spec": (
            charged_group_to_chimerax_spec(
                interaction.cation
            )
        ),
        "anion_group_spec": (
            charged_group_to_chimerax_spec(
                interaction.anion
            )
        ),
        "residue_spec": (
            salt_bridge_interaction_to_chimerax_spec(
                interaction,
                residues_only=True,
            )
        ),
        "selection_spec": (
            salt_bridge_interaction_to_chimerax_spec(
                interaction
            )
        ),
        "pseudobond_group": (
            make_salt_bridge_pseudobond_group_name(
                interaction
            )
        ),
        "color": (
            get_salt_bridge_strength_color(
                interaction
            )
        ),
    }


def build_chimerax_salt_bridge_records(
    result: SaltBridgeResult,
    *,
    valid_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build ChimeraX-oriented records for a result.

    Parameters
    ----------
    result
        Salt-bridge result.
    valid_only
        Whether invalid interactions should be excluded.

    Returns
    -------
    List[Dict[str, Any]]
        ChimeraX records.
    """

    records: List[
        Dict[str, Any]
    ] = []

    for interaction in result.interactions:
        if (
            valid_only
            and not interaction.geometry.valid
        ):
            continue

        try:
            record = (
                build_chimerax_salt_bridge_record(
                    interaction
                )
            )

        except ChimeraXSaltBridgeError:
            continue

        records.append(
            record
        )

    return records



# =============================================================================
# 18. SELF-TESTS
# =============================================================================


# =============================================================================
# 18.1. TEST INFRASTRUCTURE
# =============================================================================


@dataclass
class _MockChain:
    """
    Minimal chain-like object used by salt-bridge self-tests.
    """

    chain_id: str = "A"

    @property
    def id(self) -> str:
        """
        Return the chain identifier.
        """

        return self.chain_id


@dataclass
class _MockStructure:
    """
    Minimal structure-like object used by salt-bridge self-tests.
    """

    name: str = "mock_structure"
    id_string: str = "1"
    session: Any = None

    @property
    def id(self) -> Tuple[int]:
        """
        Return a ChimeraX-like model identifier.
        """

        try:
            model_id = int(
                str(
                    self.id_string
                ).split(".")[0]
            )

        except (
            TypeError,
            ValueError,
        ):
            model_id = 1

        return (
            model_id,
        )


@dataclass
class _MockResidue:
    """
    Minimal residue-like object used by salt-bridge self-tests.
    """

    name: str
    number: int
    chain_id: str = "A"
    atoms: List[Any] = field(
        default_factory=list
    )
    structure: Any = None
    insertion_code: str = ""

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize chain and atom ownership.
        """

        self.chain = _MockChain(
            self.chain_id
        )

        if self.structure is None:
            self.structure = _MockStructure()

        for atom in self.atoms:
            atom.residue = self
            atom.structure = self.structure

    @property
    def id(self) -> int:
        """
        Return the residue number.
        """

        return self.number

    @property
    def principal_atom(
        self,
    ) -> Any:
        """
        Return the first atom, when available.
        """

        if not self.atoms:
            return None

        return self.atoms[0]

    def add_atom(
        self,
        atom: Any,
    ) -> Any:
        """
        Add an atom and update ownership references.
        """

        atom.residue = self
        atom.structure = self.structure

        self.atoms.append(
            atom
        )

        return atom


@dataclass
class _MockAtom:
    """
    Minimal atom-like object used by salt-bridge self-tests.
    """

    name: str
    element: str
    coord: Tuple[float, float, float]
    serial_number: int = 1
    formal_charge: Optional[float] = None
    partial_charge: Optional[float] = None
    residue: Any = None
    structure: Any = None

    @property
    def coordinates(
        self,
    ) -> Tuple[float, float, float]:
        """
        Return atom coordinates.
        """

        return self.coord

    @coordinates.setter
    def coordinates(
        self,
        value: Sequence[float],
    ) -> None:
        """
        Set atom coordinates.
        """

        self.coord = _test_coordinate_tuple(
            value
        )

    @property
    def scene_coord(
        self,
    ) -> Tuple[float, float, float]:
        """
        Return a ChimeraX-like scene coordinate.
        """

        return self.coord

    @property
    def serial(
        self,
    ) -> int:
        """
        Return the atom serial number.
        """

        return self.serial_number

    @property
    def idatm_type(
        self,
    ) -> str:
        """
        Return a minimal atom-type-like value.
        """

        return self.element


@dataclass
class _MockDockModel:
    """
    Minimal DockModel-like object used by integration self-tests.
    """

    source: Any
    pose_id: Optional[
        Union[str, int]
    ] = None
    model_id: Optional[
        Union[str, int]
    ] = None

    saltbridge: List[Any] = field(
        default_factory=list
    )

    statistics: Dict[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class _SelfTestRecord:
    """
    Record describing one self-test execution.
    """

    name: str
    passed: bool
    duration_seconds: float = 0.0
    message: str = ""
    exception_type: Optional[str] = None


@dataclass
class _SelfTestReport:
    """
    Aggregated salt-bridge self-test report.
    """

    module_name: str = "saltbridge"
    module_version: str = __version__

    records: List[
        _SelfTestRecord
    ] = field(
        default_factory=list
    )

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def test_count(
        self,
    ) -> int:
        """
        Return the number of executed tests.
        """

        return len(
            self.records
        )

    @property
    def passed_count(
        self,
    ) -> int:
        """
        Return the number of passed tests.
        """

        return sum(
            1
            for record in self.records
            if record.passed
        )

    @property
    def failed_count(
        self,
    ) -> int:
        """
        Return the number of failed tests.
        """

        return (
            self.test_count
            - self.passed_count
        )

    @property
    def success(
        self,
    ) -> bool:
        """
        Return whether all tests passed.
        """

        return (
            self.test_count > 0
            and self.failed_count == 0
        )

    def add_record(
        self,
        record: _SelfTestRecord,
    ) -> None:
        """
        Add one self-test record.
        """

        if not isinstance(
            record,
            _SelfTestRecord,
        ):
            raise SaltBridgeSelfTestError(
                "record must be a _SelfTestRecord instance."
            )

        self.records.append(
            record
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert the report into a JSON-safe dictionary.
        """

        return {
            "module_name": self.module_name,
            "module_version": (
                self.module_version
            ),
            "success": self.success,
            "test_count": self.test_count,
            "passed_count": (
                self.passed_count
            ),
            "failed_count": (
                self.failed_count
            ),
            "started_at": self.started_at,
            "finished_at": (
                self.finished_at
            ),
            "records": [
                {
                    "name": record.name,
                    "passed": (
                        record.passed
                    ),
                    "duration_seconds": (
                        record.duration_seconds
                    ),
                    "message": (
                        record.message
                    ),
                    "exception_type": (
                        record.exception_type
                    ),
                }
                for record in self.records
            ],
            "metadata": make_json_safe(
                self.metadata
            ),
        }


def _test_coordinate_tuple(
    coordinate: Sequence[float],
) -> Tuple[float, float, float]:
    """
    Convert a coordinate-like value into a three-float tuple.
    """

    if coordinate is None:
        raise SaltBridgeSelfTestError(
            "A coordinate is required."
        )

    try:
        coordinate_values = tuple(
            float(value)
            for value in coordinate
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise SaltBridgeSelfTestError(
            "Invalid test coordinate."
        ) from error

    if len(
        coordinate_values
    ) != 3:
        raise SaltBridgeSelfTestError(
            "Test coordinates must contain exactly three values."
        )

    if not all(
        math.isfinite(value)
        for value in coordinate_values
    ):
        raise SaltBridgeSelfTestError(
            "Test coordinates must be finite."
        )

    return (
        coordinate_values[0],
        coordinate_values[1],
        coordinate_values[2],
    )


def _make_test_atom(
    name: str,
    element: str,
    coordinate: Sequence[float],
    *,
    serial_number: int = 1,
    formal_charge: Optional[float] = None,
    partial_charge: Optional[float] = None,
    residue: Optional[
        _MockResidue
    ] = None,
) -> _MockAtom:
    """
    Create one atom for salt-bridge self-tests.
    """

    atom = _MockAtom(
        name=str(name),
        element=str(element),
        coord=_test_coordinate_tuple(
            coordinate
        ),
        serial_number=int(
            serial_number
        ),
        formal_charge=(
            formal_charge
        ),
        partial_charge=(
            partial_charge
        ),
        residue=residue,
        structure=(
            residue.structure
            if residue is not None
            else None
        ),
    )

    if (
        residue is not None
        and atom not in residue.atoms
    ):
        residue.add_atom(
            atom
        )

    return atom


def _make_test_residue(
    name: str,
    number: int,
    *,
    chain_id: str = "A",
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create an empty residue for salt-bridge self-tests.
    """

    return _MockResidue(
        name=str(
            name
        ).upper(),
        number=int(
            number
        ),
        chain_id=str(
            chain_id
        ),
        atoms=[],
        structure=(
            structure
            if structure is not None
            else _MockStructure()
        ),
    )


def _make_test_structure(
    *,
    name: str = "mock_structure",
    model_id: Union[str, int] = 1,
) -> _MockStructure:
    """
    Create a structure-like object for self-tests.
    """

    return _MockStructure(
        name=name,
        id_string=str(
            model_id
        ),
    )


def _translate_coordinate(
    coordinate: Sequence[float],
    translation: Sequence[float],
) -> Tuple[float, float, float]:
    """
    Translate one three-dimensional coordinate.
    """

    point = _test_coordinate_tuple(
        coordinate
    )

    shift = _test_coordinate_tuple(
        translation
    )

    return (
        point[0] + shift[0],
        point[1] + shift[1],
        point[2] + shift[2],
    )


def _rotate_coordinate_z(
    coordinate: Sequence[float],
    angle_degrees: float,
    *,
    origin: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> Tuple[float, float, float]:
    """
    Rotate one coordinate around the z axis.
    """

    point = _test_coordinate_tuple(
        coordinate
    )

    rotation_origin = (
        _test_coordinate_tuple(
            origin
        )
    )

    angle_radians = math.radians(
        float(
            angle_degrees
        )
    )

    cosine = math.cos(
        angle_radians
    )

    sine = math.sin(
        angle_radians
    )

    relative_x = (
        point[0]
        - rotation_origin[0]
    )

    relative_y = (
        point[1]
        - rotation_origin[1]
    )

    rotated_x = (
        relative_x * cosine
        - relative_y * sine
    )

    rotated_y = (
        relative_x * sine
        + relative_y * cosine
    )

    return (
        rotated_x
        + rotation_origin[0],
        rotated_y
        + rotation_origin[1],
        point[2],
    )


def _transform_test_atoms(
    atoms: Iterable[_MockAtom],
    *,
    translation: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    rotation_z_degrees: float = 0.0,
    origin: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    in_place: bool = False,
) -> List[_MockAtom]:
    """
    Apply rotation and translation to test atoms.
    """

    transformed_atoms: List[
        _MockAtom
    ] = []

    for atom in atoms:
        rotated_coordinate = (
            _rotate_coordinate_z(
                atom.coord,
                rotation_z_degrees,
                origin=origin,
            )
        )

        transformed_coordinate = (
            _translate_coordinate(
                rotated_coordinate,
                translation,
            )
        )

        if in_place:
            atom.coord = (
                transformed_coordinate
            )

            transformed_atom = atom

        else:
            transformed_atom = (
                _make_test_atom(
                    atom.name,
                    atom.element,
                    transformed_coordinate,
                    serial_number=(
                        atom.serial_number
                    ),
                    formal_charge=(
                        atom.formal_charge
                    ),
                    partial_charge=(
                        atom.partial_charge
                    ),
                )
            )

        transformed_atoms.append(
            transformed_atom
        )

    return transformed_atoms


def _make_mock_lysine(
    *,
    number: int = 10,
    chain_id: str = "A",
    nz_coordinate: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal positively charged lysine residue.
    """

    residue = _make_test_residue(
        "LYS",
        number,
        chain_id=chain_id,
        structure=structure,
    )

    nz = _make_test_atom(
        "NZ",
        "N",
        nz_coordinate,
        serial_number=1,
        formal_charge=1.0,
        residue=residue,
    )

    _make_test_atom(
        "CE",
        "C",
        _translate_coordinate(
            nz_coordinate,
            (
                -1.45,
                0.0,
                0.0,
            ),
        ),
        serial_number=2,
        residue=residue,
    )

    assert nz in residue.atoms

    return residue


def _make_mock_arginine(
    *,
    number: int = 20,
    chain_id: str = "A",
    center: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal positively charged arginine residue.
    """

    residue = _make_test_residue(
        "ARG",
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "CZ",
        "C",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    _make_test_atom(
        "NE",
        "N",
        _translate_coordinate(
            center_coordinate,
            (
                -1.25,
                0.0,
                0.0,
            ),
        ),
        serial_number=2,
        partial_charge=0.33,
        residue=residue,
    )

    _make_test_atom(
        "NH1",
        "N",
        _translate_coordinate(
            center_coordinate,
            (
                0.65,
                1.05,
                0.0,
            ),
        ),
        serial_number=3,
        partial_charge=0.33,
        residue=residue,
    )

    _make_test_atom(
        "NH2",
        "N",
        _translate_coordinate(
            center_coordinate,
            (
                0.65,
                -1.05,
                0.0,
            ),
        ),
        serial_number=4,
        partial_charge=0.34,
        residue=residue,
    )

    return residue


def _make_mock_hip(
    *,
    number: int = 30,
    chain_id: str = "A",
    center: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal protonated histidine residue.
    """

    residue = _make_test_residue(
        "HIP",
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    atom_definitions = (
        (
            "CG",
            "C",
            (
                -1.0,
                0.0,
                0.0,
            ),
        ),
        (
            "ND1",
            "N",
            (
                -0.30,
                1.0,
                0.0,
            ),
        ),
        (
            "CE1",
            "C",
            (
                0.90,
                0.65,
                0.0,
            ),
        ),
        (
            "NE2",
            "N",
            (
                0.90,
                -0.65,
                0.0,
            ),
        ),
        (
            "CD2",
            "C",
            (
                -0.30,
                -1.0,
                0.0,
            ),
        ),
    )

    for serial_number, (
        atom_name,
        element,
        offset,
    ) in enumerate(
        atom_definitions,
        start=1,
    ):
        partial_charge = (
            0.50
            if atom_name in {
                "ND1",
                "NE2",
            }
            else None
        )

        _make_test_atom(
            atom_name,
            element,
            _translate_coordinate(
                center_coordinate,
                offset,
            ),
            serial_number=(
                serial_number
            ),
            partial_charge=(
                partial_charge
            ),
            residue=residue,
        )

    return residue


def _make_mock_aspartate(
    *,
    number: int = 40,
    chain_id: str = "B",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal negatively charged aspartate residue.
    """

    residue = _make_test_residue(
        "ASP",
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "CG",
        "C",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    _make_test_atom(
        "OD1",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                0.65,
                0.0,
            ),
        ),
        serial_number=2,
        partial_charge=-0.50,
        residue=residue,
    )

    _make_test_atom(
        "OD2",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                -0.65,
                0.0,
            ),
        ),
        serial_number=3,
        partial_charge=-0.50,
        residue=residue,
    )

    return residue


def _make_mock_glutamate(
    *,
    number: int = 50,
    chain_id: str = "B",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal negatively charged glutamate residue.
    """

    residue = _make_test_residue(
        "GLU",
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "CD",
        "C",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    _make_test_atom(
        "OE1",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                0.65,
                0.0,
            ),
        ),
        serial_number=2,
        partial_charge=-0.50,
        residue=residue,
    )

    _make_test_atom(
        "OE2",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                -0.65,
                0.0,
            ),
        ),
        serial_number=3,
        partial_charge=-0.50,
        residue=residue,
    )

    return residue


def _make_mock_cationic_ligand(
    *,
    residue_name: str = "LIG",
    number: int = 101,
    chain_id: str = "L",
    center: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    formal_charge: float = 1.0,
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal ligand containing a cationic nitrogen.
    """

    residue = _make_test_residue(
        residue_name,
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "N1",
        "N",
        center_coordinate,
        serial_number=1,
        formal_charge=(
            formal_charge
        ),
        residue=residue,
    )

    for serial_number, offset in enumerate(
        (
            (
                1.4,
                0.0,
                0.0,
            ),
            (
                -0.7,
                1.2,
                0.0,
            ),
            (
                -0.7,
                -1.2,
                0.0,
            ),
        ),
        start=2,
    ):
        _make_test_atom(
            f"C{serial_number - 1}",
            "C",
            _translate_coordinate(
                center_coordinate,
                offset,
            ),
            serial_number=(
                serial_number
            ),
            residue=residue,
        )

    return residue


def _make_mock_carboxylate_ligand(
    *,
    residue_name: str = "LIG",
    number: int = 102,
    chain_id: str = "L",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal ligand carboxylate.
    """

    residue = _make_test_residue(
        residue_name,
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "C1",
        "C",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    _make_test_atom(
        "O1",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                0.65,
                0.0,
            ),
        ),
        serial_number=2,
        partial_charge=-0.50,
        residue=residue,
    )

    _make_test_atom(
        "O2",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                0.0,
                -0.65,
                0.0,
            ),
        ),
        serial_number=3,
        partial_charge=-0.50,
        residue=residue,
    )

    return residue


def _make_mock_phosphate_ligand(
    *,
    residue_name: str = "LIG",
    number: int = 103,
    chain_id: str = "L",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal ligand phosphate group.
    """

    residue = _make_test_residue(
        residue_name,
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "P1",
        "P",
        center_coordinate,
        serial_number=1,
        formal_charge=0.0,
        residue=residue,
    )

    phosphate_offsets = (
        (
            1.0,
            0.0,
            0.0,
        ),
        (
            -1.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            1.0,
            0.0,
        ),
        (
            0.0,
            -1.0,
            0.0,
        ),
    )

    for serial_number, offset in enumerate(
        phosphate_offsets,
        start=2,
    ):
        _make_test_atom(
            f"O{serial_number - 1}",
            "O",
            _translate_coordinate(
                center_coordinate,
                offset,
            ),
            serial_number=(
                serial_number
            ),
            partial_charge=-0.50,
            residue=residue,
        )

    return residue


def _make_mock_sulfonate_ligand(
    *,
    residue_name: str = "LIG",
    number: int = 104,
    chain_id: str = "L",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a minimal ligand sulfonate group.
    """

    residue = _make_test_residue(
        residue_name,
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "S1",
        "S",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    sulfonate_offsets = (
        (
            1.0,
            0.0,
            0.0,
        ),
        (
            -0.5,
            0.866,
            0.0,
        ),
        (
            -0.5,
            -0.866,
            0.0,
        ),
    )

    for serial_number, offset in enumerate(
        sulfonate_offsets,
        start=2,
    ):
        _make_test_atom(
            f"O{serial_number - 1}",
            "O",
            _translate_coordinate(
                center_coordinate,
                offset,
            ),
            serial_number=(
                serial_number
            ),
            partial_charge=(
                -1.0 / 3.0
            ),
            residue=residue,
        )

    return residue


def _make_mock_neutral_ligand(
    *,
    residue_name: str = "LIG",
    number: int = 105,
    chain_id: str = "L",
    center: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    structure: Optional[
        _MockStructure
    ] = None,
) -> _MockResidue:
    """
    Create a neutral ligand used in negative tests.
    """

    residue = _make_test_residue(
        residue_name,
        number,
        chain_id=chain_id,
        structure=structure,
    )

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    _make_test_atom(
        "C1",
        "C",
        center_coordinate,
        serial_number=1,
        residue=residue,
    )

    _make_test_atom(
        "O1",
        "O",
        _translate_coordinate(
            center_coordinate,
            (
                1.3,
                0.0,
                0.0,
            ),
        ),
        serial_number=2,
        partial_charge=-0.20,
        residue=residue,
    )

    _make_test_atom(
        "N1",
        "N",
        _translate_coordinate(
            center_coordinate,
            (
                -1.3,
                0.0,
                0.0,
            ),
        ),
        serial_number=3,
        partial_charge=0.20,
        residue=residue,
    )

    return residue


def _make_test_source(
    residues: Iterable[
        _MockResidue
    ],
) -> List[_MockResidue]:
    """
    Materialize a residue collection used as a test source.
    """

    return list(
        residues
    )


def _make_test_charged_atom(
    *,
    name: str,
    element: str,
    coordinate: Sequence[float],
    polarity: str,
    effective_charge: float,
    source: str = "self_test",
    residue: Optional[
        _MockResidue
    ] = None,
    serial_number: int = 1,
) -> ChargedAtom:
    """
    Create a ChargedAtom dataclass for direct tests.
    """

    atom = _make_test_atom(
        name,
        element,
        coordinate,
        serial_number=(
            serial_number
        ),
        formal_charge=(
            effective_charge
        ),
        residue=residue,
    )

    return ChargedAtom(
        atom=atom,
        residue=residue,
        name=name,
        element=element,
        coordinate=(
            _test_coordinate_tuple(
                coordinate
            )
        ),
        formal_charge=(
            effective_charge
        ),
        partial_charge=None,
        effective_charge=(
            effective_charge
        ),
        polarity=polarity,
        source=source,
        metadata={
            "self_test": True,
        },
    )


def _make_test_charged_group(
    *,
    group_id: str,
    group_type: str,
    polarity: str,
    center: Sequence[float],
    net_charge: float,
    atom_names: Optional[
        Sequence[str]
    ] = None,
    residue: Optional[
        _MockResidue
    ] = None,
    confidence: float = 1.0,
    source: str = "self_test",
) -> ChargedGroup:
    """
    Create a ChargedGroup dataclass for direct tests.
    """

    center_coordinate = (
        _test_coordinate_tuple(
            center
        )
    )

    if atom_names is None:
        atom_names = (
            "N1",
        ) if polarity == "positive" else (
            "O1",
        )

    charged_atoms: List[
        ChargedAtom
    ] = []

    atom_count = len(
        atom_names
    )

    charge_per_atom = (
        net_charge / atom_count
        if atom_count > 0
        else net_charge
    )

    for atom_index, atom_name in enumerate(
        atom_names,
        start=1,
    ):
        angle = (
            2.0
            * math.pi
            * (
                atom_index - 1
            )
            / max(
                atom_count,
                1,
            )
        )

        coordinate = (
            center_coordinate[0]
            + 0.30
            * math.cos(
                angle
            ),
            center_coordinate[1]
            + 0.30
            * math.sin(
                angle
            ),
            center_coordinate[2],
        )

        element = (
            "N"
            if polarity == "positive"
            else "O"
        )

        charged_atoms.append(
            _make_test_charged_atom(
                name=atom_name,
                element=element,
                coordinate=coordinate,
                polarity=polarity,
                effective_charge=(
                    charge_per_atom
                ),
                source=source,
                residue=residue,
                serial_number=(
                    atom_index
                ),
            )
        )

    representative_atom = (
        charged_atoms[0].atom
        if charged_atoms
        else None
    )

    return ChargedGroup(
        group_id=group_id,
        group_type=group_type,
        polarity=polarity,
        atoms=charged_atoms,
        residue=residue,
        center=center_coordinate,
        net_charge=net_charge,
        representative_atom=(
            representative_atom
        ),
        source=source,
        confidence=confidence,
        metadata={
            "self_test": True,
        },
    )


def _make_test_cation_group(
    *,
    group_id: str = "test_cation",
    center: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    group_type: str = "ammonium",
    residue: Optional[
        _MockResidue
    ] = None,
    net_charge: float = 1.0,
) -> ChargedGroup:
    """
    Create a positive charged group for tests.
    """

    return _make_test_charged_group(
        group_id=group_id,
        group_type=group_type,
        polarity="positive",
        center=center,
        net_charge=net_charge,
        atom_names=(
            "N1",
        ),
        residue=residue,
    )


def _make_test_anion_group(
    *,
    group_id: str = "test_anion",
    center: Sequence[float] = (
        3.0,
        0.0,
        0.0,
    ),
    group_type: str = "carboxylate",
    residue: Optional[
        _MockResidue
    ] = None,
    net_charge: float = -1.0,
) -> ChargedGroup:
    """
    Create a negative charged group for tests.
    """

    return _make_test_charged_group(
        group_id=group_id,
        group_type=group_type,
        polarity="negative",
        center=center,
        net_charge=net_charge,
        atom_names=(
            "O1",
            "O2",
        ),
        residue=residue,
    )


def _make_test_geometry(
    *,
    center_distance: float = 3.0,
    minimum_atom_distance: float = 2.7,
    maximum_atom_distance: float = 3.3,
    mean_atom_distance: float = 3.0,
    contact_count: int = 1,
    valid: bool = True,
    rejection_reason: Optional[str] = None,
    positive_atom: Any = None,
    negative_atom: Any = None,
) -> SaltBridgeGeometry:
    """
    Create SaltBridgeGeometry for direct tests.
    """

    return SaltBridgeGeometry(
        center_distance=float(
            center_distance
        ),
        minimum_atom_distance=float(
            minimum_atom_distance
        ),
        maximum_atom_distance=float(
            maximum_atom_distance
        ),
        mean_atom_distance=float(
            mean_atom_distance
        ),
        contact_count=int(
            contact_count
        ),
        closest_positive_atom=(
            positive_atom
        ),
        closest_negative_atom=(
            negative_atom
        ),
        valid=bool(
            valid
        ),
        rejection_reason=(
            rejection_reason
        ),
    )


def _make_test_interaction(
    *,
    interaction_id: str = (
        "salt_bridge_test_001"
    ),
    cation: Optional[
        ChargedGroup
    ] = None,
    anion: Optional[
        ChargedGroup
    ] = None,
    geometry: Optional[
        SaltBridgeGeometry
    ] = None,
    interaction_type: str = SALT_BRIDGE,
    strength: str = STRENGTH_STRONG,
    score: float = 1.0,
    pose_id: Optional[
        Union[str, int]
    ] = 1,
    model_id: Optional[
        Union[str, int]
    ] = "model_1",
) -> SaltBridgeInteraction:
    """
    Create SaltBridgeInteraction for direct tests.
    """

    resolved_cation = (
        cation
        if cation is not None
        else _make_test_cation_group()
    )

    resolved_anion = (
        anion
        if anion is not None
        else _make_test_anion_group()
    )

    if geometry is None:
        positive_atom = (
            resolved_cation
            .representative_atom
        )

        negative_atom = (
            resolved_anion
            .representative_atom
        )

        geometry = _make_test_geometry(
            positive_atom=(
                positive_atom
            ),
            negative_atom=(
                negative_atom
            ),
        )

    return SaltBridgeInteraction(
        interaction_id=interaction_id,
        interaction_type=(
            interaction_type
        ),
        cation=resolved_cation,
        anion=resolved_anion,
        geometry=geometry,
        strength=strength,
        score=float(
            score
        ),
        pose_id=pose_id,
        model_id=model_id,
        metadata={
            "self_test": True,
        },
    )


def _make_test_result(
    *,
    interactions: Optional[
        Iterable[
            SaltBridgeInteraction
        ]
    ] = None,
    cationic_groups: Optional[
        Iterable[ChargedGroup]
    ] = None,
    anionic_groups: Optional[
        Iterable[ChargedGroup]
    ] = None,
    pose_id: Optional[
        Union[str, int]
    ] = 1,
    model_id: Optional[
        Union[str, int]
    ] = "model_1",
) -> SaltBridgeResult:
    """
    Create SaltBridgeResult for direct tests.
    """

    interaction_list = (
        list(
            interactions
        )
        if interactions is not None
        else [
            _make_test_interaction(
                pose_id=pose_id,
                model_id=model_id,
            )
        ]
    )

    if cationic_groups is None:
        cationic_group_list = (
            unique_preserve_order(
                interaction.cation
                for interaction
                in interaction_list
            )
        )

    else:
        cationic_group_list = list(
            cationic_groups
        )

    if anionic_groups is None:
        anionic_group_list = (
            unique_preserve_order(
                interaction.anion
                for interaction
                in interaction_list
            )
        )

    else:
        anionic_group_list = list(
            anionic_groups
        )

    return SaltBridgeResult(
        interactions=interaction_list,
        cationic_groups=list(
            cationic_group_list
        ),
        anionic_groups=list(
            anionic_group_list
        ),
        statistics={},
        pose_id=pose_id,
        model_id=model_id,
        warnings=[],
        metadata={
            "self_test": True,
        },
    )


def _make_test_pose_results(
    *,
    pose_count: int = 3,
) -> List[SaltBridgeResult]:
    """
    Create multiple pose results for persistence and ranking tests.
    """

    if pose_count < 1:
        raise SaltBridgeSelfTestError(
            "pose_count must be at least one."
        )

    results: List[
        SaltBridgeResult
    ] = []

    for pose_index in range(
        1,
        pose_count + 1,
    ):
        cation_residue = (
            _make_mock_lysine(
                number=10,
                chain_id="A",
            )
        )

        anion_residue = (
            _make_mock_aspartate(
                number=40,
                chain_id="B",
                center=(
                    2.8
                    + 0.15
                    * pose_index,
                    0.0,
                    0.0,
                ),
            )
        )

        cation_group = (
            _make_test_cation_group(
                group_id=(
                    f"cation_pose_{pose_index}"
                ),
                center=(
                    0.0,
                    0.0,
                    0.0,
                ),
                residue=(
                    cation_residue
                ),
            )
        )

        anion_group = (
            _make_test_anion_group(
                group_id=(
                    f"anion_pose_{pose_index}"
                ),
                center=(
                    2.8
                    + 0.15
                    * pose_index,
                    0.0,
                    0.0,
                ),
                residue=(
                    anion_residue
                ),
            )
        )

        score = (
            1.0
            + 0.25
            * pose_index
        )

        interaction = (
            _make_test_interaction(
                interaction_id=(
                    "persistent_salt_bridge_"
                    f"{pose_index}"
                ),
                cation=(
                    cation_group
                ),
                anion=(
                    anion_group
                ),
                score=score,
                pose_id=(
                    pose_index
                ),
                model_id=(
                    f"model_{pose_index}"
                ),
            )
        )

        results.append(
            _make_test_result(
                interactions=[
                    interaction
                ],
                pose_id=pose_index,
                model_id=(
                    f"model_{pose_index}"
                ),
            )
        )

    return results


def _assert_true(
    condition: Any,
    message: str = (
        "Expected condition to be true."
    ),
) -> None:
    """
    Assert that a condition evaluates to true.
    """

    if not condition:
        raise SaltBridgeSelfTestError(
            message
        )


def _assert_false(
    condition: Any,
    message: str = (
        "Expected condition to be false."
    ),
) -> None:
    """
    Assert that a condition evaluates to false.
    """

    if condition:
        raise SaltBridgeSelfTestError(
            message
        )


def _assert_equal(
    actual: Any,
    expected: Any,
    message: Optional[str] = None,
) -> None:
    """
    Assert strict equality.
    """

    if actual != expected:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Values are not equal: "
                f"actual={actual!r}, "
                f"expected={expected!r}."
            )
        )


def _assert_not_equal(
    actual: Any,
    unexpected: Any,
    message: Optional[str] = None,
) -> None:
    """
    Assert inequality.
    """

    if actual == unexpected:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Values unexpectedly match: "
                f"value={actual!r}."
            )
        )


def _assert_is_none(
    value: Any,
    message: str = (
        "Expected value to be None."
    ),
) -> None:
    """
    Assert that a value is None.
    """

    if value is not None:
        raise SaltBridgeSelfTestError(
            message
        )


def _assert_is_not_none(
    value: Any,
    message: str = (
        "Expected a non-None value."
    ),
) -> None:
    """
    Assert that a value is not None.
    """

    if value is None:
        raise SaltBridgeSelfTestError(
            message
        )


def _assert_is_instance(
    value: Any,
    expected_type: Any,
    message: Optional[str] = None,
) -> None:
    """
    Assert an object's type.
    """

    if not isinstance(
        value,
        expected_type,
    ):
        raise SaltBridgeSelfTestError(
            message
            or (
                "Unexpected object type: "
                f"actual={type(value).__name__}, "
                f"expected={expected_type}."
            )
        )


def _assert_length(
    value: Any,
    expected_length: int,
    message: Optional[str] = None,
) -> None:
    """
    Assert collection length.
    """

    try:
        actual_length = len(
            value
        )

    except TypeError as error:
        raise SaltBridgeSelfTestError(
            "Object has no measurable length."
        ) from error

    if actual_length != expected_length:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Unexpected collection length: "
                f"actual={actual_length}, "
                f"expected={expected_length}."
            )
        )


def _assert_empty(
    value: Any,
    message: str = (
        "Expected an empty value."
    ),
) -> None:
    """
    Assert an empty collection.
    """

    _assert_length(
        value,
        0,
        message,
    )


def _assert_not_empty(
    value: Any,
    message: str = (
        "Expected a non-empty value."
    ),
) -> None:
    """
    Assert a non-empty collection.
    """

    try:
        value_length = len(
            value
        )

    except TypeError as error:
        raise SaltBridgeSelfTestError(
            "Object has no measurable length."
        ) from error

    if value_length == 0:
        raise SaltBridgeSelfTestError(
            message
        )


def _assert_almost_equal(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-6,
    message: Optional[str] = None,
) -> None:
    """
    Assert approximate numeric equality.
    """

    actual_value = safe_float(
        actual
    )

    expected_value = safe_float(
        expected
    )

    if (
        actual_value is None
        or expected_value is None
    ):
        raise SaltBridgeSelfTestError(
            message
            or (
                "Approximate comparison requires "
                "numeric values."
            )
        )

    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise SaltBridgeSelfTestError(
            message
            or (
                "Numeric values differ: "
                f"actual={actual_value}, "
                f"expected={expected_value}, "
                f"tolerance={tolerance}."
            )
        )


def _assert_sequence_almost_equal(
    actual: Sequence[Any],
    expected: Sequence[Any],
    *,
    tolerance: float = 1e-6,
    message: Optional[str] = None,
) -> None:
    """
    Assert approximate equality between numeric sequences.
    """

    actual_values = list(
        actual
    )

    expected_values = list(
        expected
    )

    if len(
        actual_values
    ) != len(
        expected_values
    ):
        raise SaltBridgeSelfTestError(
            message
            or (
                "Sequences have different lengths."
            )
        )

    for index, (
        actual_value,
        expected_value,
    ) in enumerate(
        zip(
            actual_values,
            expected_values,
        )
    ):
        _assert_almost_equal(
            actual_value,
            expected_value,
            tolerance=tolerance,
            message=(
                message
                or (
                    "Sequence values differ "
                    f"at index {index}."
                )
            ),
        )


def _assert_between(
    value: Any,
    minimum: float,
    maximum: float,
    *,
    inclusive: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a numeric value lies within a range.
    """

    numeric_value = safe_float(
        value
    )

    if numeric_value is None:
        raise SaltBridgeSelfTestError(
            "Range assertion requires a numeric value."
        )

    if inclusive:
        valid = (
            minimum
            <= numeric_value
            <= maximum
        )

    else:
        valid = (
            minimum
            < numeric_value
            < maximum
        )

    if not valid:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Value lies outside expected range: "
                f"value={numeric_value}, "
                f"minimum={minimum}, "
                f"maximum={maximum}."
            )
        )


def _assert_contains(
    container: Any,
    expected_item: Any,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a container includes an item.
    """

    if expected_item not in container:
        raise SaltBridgeSelfTestError(
            message
            or (
                f"Expected item {expected_item!r} "
                "was not found."
            )
        )


def _assert_mapping_contains_keys(
    mapping: Mapping[str, Any],
    expected_keys: Iterable[str],
    *,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a mapping contains all expected keys.
    """

    if not isinstance(
        mapping,
        Mapping,
    ):
        raise SaltBridgeSelfTestError(
            "Expected a mapping."
        )

    missing_keys = [
        key
        for key in expected_keys
        if key not in mapping
    ]

    if missing_keys:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Mapping is missing keys: "
                + ", ".join(
                    repr(key)
                    for key in missing_keys
                )
            )
        )


def _assert_raises(
    expected_exception: Any,
    callable_object: Callable[
        ...,
        Any,
    ],
    *args: Any,
    **kwargs: Any,
) -> BaseException:
    """
    Assert that a callable raises the expected exception.
    """

    try:
        callable_object(
            *args,
            **kwargs,
        )

    except expected_exception as error:
        return error

    except Exception as error:
        raise SaltBridgeSelfTestError(
            "Unexpected exception type: "
            f"actual={type(error).__name__}, "
            f"expected={expected_exception}."
        ) from error

    raise SaltBridgeSelfTestError(
        "Expected exception was not raised: "
        f"{expected_exception}."
    )


def _assert_json_serializable(
    value: Any,
    *,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a value can be serialized as strict JSON.
    """

    try:
        json.dumps(
            value,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise SaltBridgeSelfTestError(
            message
            or (
                "Value is not strictly JSON serializable."
            )
        ) from error


def _assert_valid_charged_group(
    group: ChargedGroup,
    *,
    expected_polarity: Optional[
        str
    ] = None,
) -> None:
    """
    Assert basic ChargedGroup invariants.
    """

    _assert_is_instance(
        group,
        ChargedGroup,
    )

    _assert_not_empty(
        group.atoms,
        "Charged group must contain atoms.",
    )

    _assert_is_not_none(
        group.center,
        "Charged group must have a center.",
    )

    _assert_equal(
        len(
            group.center
        ),
        3,
        "Charged-group center must be three-dimensional.",
    )

    if expected_polarity is not None:
        _assert_equal(
            group.polarity,
            expected_polarity,
        )

    if group.polarity == "positive":
        _assert_true(
            group.net_charge > 0.0,
            "Positive groups must have positive net charge.",
        )

    elif group.polarity == "negative":
        _assert_true(
            group.net_charge < 0.0,
            "Negative groups must have negative net charge.",
        )

    else:
        raise SaltBridgeSelfTestError(
            "Unsupported charged-group polarity."
        )


def _assert_valid_geometry(
    geometry: SaltBridgeGeometry,
    *,
    expected_valid: Optional[
        bool
    ] = None,
) -> None:
    """
    Assert basic SaltBridgeGeometry invariants.
    """

    _assert_is_instance(
        geometry,
        SaltBridgeGeometry,
    )

    for value_name in (
        "center_distance",
        "minimum_atom_distance",
        "maximum_atom_distance",
        "mean_atom_distance",
    ):
        value = get_value(
            geometry,
            value_name,
            None,
        )

        _assert_is_not_none(
            value,
            (
                f"Geometry field {value_name!r} "
                "must not be None."
            ),
        )

        _assert_true(
            float(
                value
            ) >= 0.0,
            (
                f"Geometry field {value_name!r} "
                "must not be negative."
            ),
        )

    _assert_true(
        geometry.minimum_atom_distance
        <= geometry.maximum_atom_distance,
        (
            "Minimum atom distance cannot exceed "
            "maximum atom distance."
        ),
    )

    _assert_true(
        geometry.contact_count >= 0,
        "Contact count cannot be negative.",
    )

    if expected_valid is not None:
        _assert_equal(
            geometry.valid,
            expected_valid,
        )


def _assert_valid_interaction(
    interaction: SaltBridgeInteraction,
    *,
    expected_valid: Optional[
        bool
    ] = None,
) -> None:
    """
    Assert basic SaltBridgeInteraction invariants.
    """

    _assert_is_instance(
        interaction,
        SaltBridgeInteraction,
    )

    _assert_valid_charged_group(
        interaction.cation,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_valid_charged_group(
        interaction.anion,
        expected_polarity=(
            "negative"
        ),
    )

    _assert_valid_geometry(
        interaction.geometry,
        expected_valid=(
            expected_valid
        ),
    )

    _assert_true(
        interaction.score >= 0.0,
        "Interaction score cannot be negative.",
    )

    _assert_is_not_none(
        interaction.interaction_id,
        "Interaction identifier is required.",
    )


def _assert_valid_result(
    result: SaltBridgeResult,
) -> None:
    """
    Assert basic SaltBridgeResult invariants.
    """

    _assert_is_instance(
        result,
        SaltBridgeResult,
    )

    _assert_is_instance(
        result.interactions,
        list,
    )

    _assert_is_instance(
        result.cationic_groups,
        list,
    )

    _assert_is_instance(
        result.anionic_groups,
        list,
    )

    _assert_is_instance(
        result.statistics,
        Mapping,
    )

    _assert_is_instance(
        result.metadata,
        Mapping,
    )

    for interaction in (
        result.interactions
    ):
        _assert_valid_interaction(
            interaction
        )


def _run_self_test_case(
    test_name: str,
    test_callable: Callable[
        [],
        Any,
    ],
    *,
    raise_on_failure: bool = False,
) -> _SelfTestRecord:
    """
    Execute one self-test and return its record.
    """

    import time

    start_time = (
        time.perf_counter()
    )

    try:
        test_callable()

    except Exception as error:
        duration = (
            time.perf_counter()
            - start_time
        )

        record = _SelfTestRecord(
            name=test_name,
            passed=False,
            duration_seconds=(
                duration
            ),
            message=str(
                error
            ),
            exception_type=(
                type(
                    error
                ).__name__
            ),
        )

        if raise_on_failure:
            raise SaltBridgeSelfTestError(
                f"Self-test failed: {test_name}: {error}"
            ) from error

        return record

    duration = (
        time.perf_counter()
        - start_time
    )

    return _SelfTestRecord(
        name=test_name,
        passed=True,
        duration_seconds=duration,
        message="passed",
        exception_type=None,
    )


def _run_self_test_group(
    tests: Iterable[
        Tuple[
            str,
            Callable[
                [],
                Any,
            ],
        ]
    ],
    *,
    report: Optional[
        _SelfTestReport
    ] = None,
    raise_on_failure: bool = False,
) -> _SelfTestReport:
    """
    Execute a group of named self-tests.
    """

    resolved_report = (
        report
        if report is not None
        else _SelfTestReport()
    )

    for test_name, test_callable in tests:
        test_record = (
            _run_self_test_case(
                test_name,
                test_callable,
                raise_on_failure=(
                    raise_on_failure
                ),
            )
        )

        resolved_report.add_record(
            test_record
        )

    return resolved_report


def _format_self_test_record(
    record: _SelfTestRecord,
) -> str:
    """
    Format one self-test record.
    """

    status = (
        "PASS"
        if record.passed
        else "FAIL"
    )

    line = (
        f"[{status}] {record.name} "
        f"({record.duration_seconds:.6f} s)"
    )

    if (
        not record.passed
        and record.message
    ):
        line += (
            f": {record.message}"
        )

    return line


def _format_self_test_report(
    report: _SelfTestReport,
    *,
    include_records: bool = True,
) -> str:
    """
    Format a complete self-test report.
    """

    lines = [
        (
            "DockAnalyzer saltbridge.py "
            "self-test report"
        ),
        (
            f"Module version: "
            f"{report.module_version}"
        ),
        (
            f"Tests: {report.test_count}"
        ),
        (
            f"Passed: {report.passed_count}"
        ),
        (
            f"Failed: {report.failed_count}"
        ),
        (
            "Status: "
            + (
                "SUCCESS"
                if report.success
                else "FAILURE"
            )
        ),
    ]

    if include_records:
        lines.append(
            ""
        )

        lines.extend(
            _format_self_test_record(
                record
            )
            for record in report.records
        )

    return "\n".join(
        lines
    )


def _print_self_test_report(
    report: _SelfTestReport,
    *,
    include_records: bool = True,
) -> None:
    """
    Print a salt-bridge self-test report.
    """

    print(
        _format_self_test_report(
            report,
            include_records=(
                include_records
            ),
        )
    )


def _test_infrastructure_smoke_test() -> None:
    """
    Verify the fundamental self-test infrastructure.
    """

    structure = _make_test_structure(
        model_id=1
    )

    lysine = _make_mock_lysine(
        structure=structure
    )

    aspartate = _make_mock_aspartate(
        structure=structure
    )

    _assert_equal(
        lysine.name,
        "LYS",
    )

    _assert_equal(
        aspartate.name,
        "ASP",
    )

    _assert_not_empty(
        lysine.atoms
    )

    _assert_not_empty(
        aspartate.atoms
    )

    cation = _make_test_cation_group(
        residue=lysine
    )

    anion = _make_test_anion_group(
        residue=aspartate
    )

    _assert_valid_charged_group(
        cation,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_valid_charged_group(
        anion,
        expected_polarity=(
            "negative"
        ),
    )

    interaction = _make_test_interaction(
        cation=cation,
        anion=anion,
    )

    _assert_valid_interaction(
        interaction,
        expected_valid=True,
    )

    result = _make_test_result(
        interactions=[
            interaction
        ]
    )

    _assert_valid_result(
        result
    )

    serialized_result = (
        salt_bridge_result_to_dict(
            result
        )
    )

    _assert_json_serializable(
        serialized_result
    )


def run_salt_bridge_test_infrastructure_smoke_test(
    *,
    raise_on_failure: bool = True,
) -> _SelfTestRecord:
    """
    Run the Section 18.1 infrastructure smoke test.
    """

    return _run_self_test_case(
        "18.1.test_infrastructure_smoke_test",
        _test_infrastructure_smoke_test,
        raise_on_failure=(
            raise_on_failure
        ),
    )



# =============================================================================
# 18.2. RECOGNITION AND GEOMETRY TESTS
# =============================================================================


# =============================================================================
# 18.2.1. TEST FUNCTION RESOLUTION
# =============================================================================


def _resolve_self_test_callable(
    *function_names: str,
) -> Callable[..., Any]:
    """
    Resolve a module-level callable by one of several possible names.

    This helper allows the self-tests to tolerate minor naming differences
    between DockAnalyzer module versions.

    Parameters
    ----------
    *function_names
        Candidate function names.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.

    Raises
    ------
    SaltBridgeSelfTestError
        If none of the candidate functions exists.
    """

    module_globals = globals()

    for function_name in function_names:
        candidate = module_globals.get(
            function_name
        )

        if callable(
            candidate
        ):
            return candidate

    raise SaltBridgeSelfTestError(
        "Could not resolve any expected function: "
        + ", ".join(
            function_names
        )
    )


def _call_recognize_cationic_groups(
    source: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> List[ChargedGroup]:
    """
    Call the available cationic-group recognition function.
    """

    recognition_function = (
        _resolve_self_test_callable(
            "recognize_cationic_groups",
            "find_cationic_groups",
            "identify_cationic_groups",
            "detect_cationic_groups",
        )
    )

    try:
        result = recognition_function(
            source,
            config=config,
        )

    except TypeError:
        try:
            result = recognition_function(
                source,
                config,
            )

        except TypeError:
            result = recognition_function(
                source
            )

    return list(
        result
        or []
    )


def _call_recognize_anionic_groups(
    source: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> List[ChargedGroup]:
    """
    Call the available anionic-group recognition function.
    """

    recognition_function = (
        _resolve_self_test_callable(
            "recognize_anionic_groups",
            "find_anionic_groups",
            "identify_anionic_groups",
            "detect_anionic_groups",
        )
    )

    try:
        result = recognition_function(
            source,
            config=config,
        )

    except TypeError:
        try:
            result = recognition_function(
                source,
                config,
            )

        except TypeError:
            result = recognition_function(
                source
            )

    return list(
        result
        or []
    )


def _call_recognize_charged_groups(
    source: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> Tuple[
    List[ChargedGroup],
    List[ChargedGroup],
]:
    """
    Recognize positive and negative charged groups.

    The combined recognition function is preferred when available.
    Otherwise, positive and negative recognizers are called separately.
    """

    combined_function_names = (
        "recognize_charged_groups",
        "find_charged_groups",
        "identify_charged_groups",
    )

    combined_function = None

    for function_name in (
        combined_function_names
    ):
        candidate = globals().get(
            function_name
        )

        if callable(
            candidate
        ):
            combined_function = candidate
            break

    if combined_function is not None:
        try:
            result = combined_function(
                source,
                config=config,
            )

        except TypeError:
            try:
                result = combined_function(
                    source,
                    config,
                )

            except TypeError:
                result = combined_function(
                    source
                )

        if isinstance(
            result,
            Mapping,
        ):
            cationic_groups = (
                result.get(
                    "cationic_groups"
                )
                or result.get(
                    "positive"
                )
                or result.get(
                    "cations"
                )
                or []
            )

            anionic_groups = (
                result.get(
                    "anionic_groups"
                )
                or result.get(
                    "negative"
                )
                or result.get(
                    "anions"
                )
                or []
            )

            return (
                list(
                    cationic_groups
                ),
                list(
                    anionic_groups
                ),
            )

        if (
            isinstance(
                result,
                tuple,
            )
            and len(result) == 2
        ):
            return (
                list(
                    result[0]
                    or []
                ),
                list(
                    result[1]
                    or []
                ),
            )

        if isinstance(
            result,
            Iterable,
        ):
            all_groups = list(
                result
            )

            return (
                [
                    group
                    for group in all_groups
                    if group.polarity
                    == "positive"
                ],
                [
                    group
                    for group in all_groups
                    if group.polarity
                    == "negative"
                ],
            )

    return (
        _call_recognize_cationic_groups(
            source,
            config,
        ),
        _call_recognize_anionic_groups(
            source,
            config,
        ),
    )


def _call_calculate_group_center(
    group_or_atoms: Any,
) -> Tuple[float, float, float]:
    """
    Call the available charged-group center function.
    """

    center_function = (
        _resolve_self_test_callable(
            "calculate_charged_group_center",
            "calculate_group_center",
            "calculate_charge_center",
            "charged_group_center",
        )
    )

    center = center_function(
        group_or_atoms
    )

    return _test_coordinate_tuple(
        center
    )


def _call_calculate_salt_bridge_geometry(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> SaltBridgeGeometry:
    """
    Call the available salt-bridge geometry function.
    """

    geometry_function = (
        _resolve_self_test_callable(
            "calculate_salt_bridge_geometry",
            "compute_salt_bridge_geometry",
            "evaluate_salt_bridge_geometry",
            "measure_salt_bridge_geometry",
        )
    )

    try:
        geometry = geometry_function(
            cation,
            anion,
            config=config,
        )

    except TypeError:
        try:
            geometry = geometry_function(
                cation,
                anion,
                config,
            )

        except TypeError:
            geometry = geometry_function(
                cation,
                anion,
            )

    _assert_is_instance(
        geometry,
        SaltBridgeGeometry,
    )

    return geometry


def _call_calculate_group_distance(
    cation: ChargedGroup,
    anion: ChargedGroup,
) -> float:
    """
    Calculate or resolve the center-to-center group distance.
    """

    candidate_names = (
        "calculate_charged_group_distance",
        "calculate_group_distance",
        "charged_group_distance",
        "calculate_center_distance",
    )

    for function_name in candidate_names:
        candidate = globals().get(
            function_name
        )

        if callable(
            candidate
        ):
            distance = candidate(
                cation,
                anion,
            )

            normalized_distance = (
                safe_float(
                    distance
                )
            )

            if normalized_distance is None:
                raise SaltBridgeSelfTestError(
                    "Group-distance function returned "
                    "a non-numeric value."
                )

            return normalized_distance

    cation_center = (
        _test_coordinate_tuple(
            cation.center
        )
    )

    anion_center = (
        _test_coordinate_tuple(
            anion.center
        )
    )

    return math.dist(
        cation_center,
        anion_center,
    )


# =============================================================================
# 18.2.2. RECOGNITION TEST UTILITIES
# =============================================================================


def _find_group_by_residue_name(
    groups: Iterable[ChargedGroup],
    residue_name: str,
) -> Optional[ChargedGroup]:
    """
    Return the first group matching a residue name.
    """

    normalized_residue_name = str(
        residue_name
    ).strip().upper()

    for group in groups:
        residue = group.residue

        if residue is None:
            continue

        if (
            get_residue_name(
                residue
            )
            == normalized_residue_name
        ):
            return group

    return None


def _find_group_by_type_fragment(
    groups: Iterable[ChargedGroup],
    fragment: str,
) -> Optional[ChargedGroup]:
    """
    Return the first group whose type contains a text fragment.
    """

    normalized_fragment = (
        normalize_text(
            fragment,
            default="",
            lowercase=True,
        )
    )

    for group in groups:
        group_type = normalize_text(
            group.group_type,
            default="",
            lowercase=True,
        )

        if (
            normalized_fragment
            in group_type
        ):
            return group

    return None


def _assert_unique_group_ids(
    groups: Iterable[ChargedGroup],
) -> None:
    """
    Assert that recognized groups have unique identifiers.
    """

    group_list = list(
        groups
    )

    group_ids = [
        group.group_id
        for group in group_list
    ]

    _assert_equal(
        len(group_ids),
        len(
            set(
                group_ids
            )
        ),
        "Recognized charged-group identifiers must be unique.",
    )


def _assert_group_contains_atom_names(
    group: ChargedGroup,
    expected_atom_names: Iterable[str],
) -> None:
    """
    Assert that a charged group contains expected atom names.
    """

    observed_names = {
        normalize_text(
            charged_atom.name,
            default="",
            lowercase=False,
        ).upper()
        for charged_atom in group.atoms
    }

    for expected_name in (
        expected_atom_names
    ):
        _assert_contains(
            observed_names,
            str(
                expected_name
            ).upper(),
            (
                f"Expected atom {expected_name!r} "
                f"in group {group.group_id!r}."
            ),
        )


# =============================================================================
# 18.2.3. PROTEIN CATION RECOGNITION TESTS
# =============================================================================


def _test_recognize_lysine_cation() -> None:
    """
    Test recognition of the lysine terminal ammonium group.
    """

    lysine = _make_mock_lysine(
        number=10,
        chain_id="A",
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
    )

    groups = (
        _call_recognize_cationic_groups(
            _make_test_source(
                [
                    lysine
                ]
            )
        )
    )

    _assert_not_empty(
        groups,
        "Lysine should produce a cationic group.",
    )

    lysine_group = (
        _find_group_by_residue_name(
            groups,
            "LYS",
        )
    )

    _assert_is_not_none(
        lysine_group,
        "No lysine cationic group was recognized.",
    )

    _assert_valid_charged_group(
        lysine_group,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_group_contains_atom_names(
        lysine_group,
        [
            "NZ"
        ],
    )

    _assert_true(
        lysine_group.net_charge > 0.0
    )


def _test_recognize_arginine_cation() -> None:
    """
    Test recognition of an arginine guanidinium group.
    """

    arginine = _make_mock_arginine(
        number=20,
        chain_id="A",
        center=(
            0.0,
            0.0,
            0.0,
        ),
    )

    groups = (
        _call_recognize_cationic_groups(
            [
                arginine
            ]
        )
    )

    arginine_group = (
        _find_group_by_residue_name(
            groups,
            "ARG",
        )
    )

    _assert_is_not_none(
        arginine_group,
        "No arginine cationic group was recognized.",
    )

    _assert_valid_charged_group(
        arginine_group,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_group_contains_atom_names(
        arginine_group,
        [
            "NE",
            "NH1",
            "NH2",
        ],
    )

    normalized_type = normalize_text(
        arginine_group.group_type,
        default="",
        lowercase=True,
    )

    _assert_true(
        (
            "guanid"
            in normalized_type
            or "argin"
            in normalized_type
        ),
        "Arginine group should be classified as guanidinium-like.",
    )


def _test_recognize_protonated_histidine_cation() -> None:
    """
    Test recognition of protonated histidine.
    """

    histidine = _make_mock_hip(
        number=30,
        chain_id="A",
    )

    groups = (
        _call_recognize_cationic_groups(
            [
                histidine
            ]
        )
    )

    histidine_group = (
        _find_group_by_residue_name(
            groups,
            "HIP",
        )
    )

    _assert_is_not_none(
        histidine_group,
        "Protonated histidine should be recognized as cationic.",
    )

    _assert_valid_charged_group(
        histidine_group,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_group_contains_atom_names(
        histidine_group,
        [
            "ND1",
            "NE2",
        ],
    )


def _test_recognize_multiple_protein_cations() -> None:
    """
    Test simultaneous recognition of multiple protein cations.
    """

    residues = [
        _make_mock_lysine(
            number=10
        ),
        _make_mock_arginine(
            number=20
        ),
        _make_mock_hip(
            number=30
        ),
    ]

    groups = (
        _call_recognize_cationic_groups(
            residues
        )
    )

    recognized_residue_names = {
        get_residue_name(
            group.residue
        )
        for group in groups
        if group.residue is not None
    }

    _assert_contains(
        recognized_residue_names,
        "LYS",
    )

    _assert_contains(
        recognized_residue_names,
        "ARG",
    )

    _assert_contains(
        recognized_residue_names,
        "HIP",
    )

    _assert_unique_group_ids(
        groups
    )


# =============================================================================
# 18.2.4. PROTEIN ANION RECOGNITION TESTS
# =============================================================================


def _test_recognize_aspartate_anion() -> None:
    """
    Test recognition of an aspartate carboxylate.
    """

    aspartate = _make_mock_aspartate(
        number=40,
        chain_id="B",
    )

    groups = (
        _call_recognize_anionic_groups(
            [
                aspartate
            ]
        )
    )

    aspartate_group = (
        _find_group_by_residue_name(
            groups,
            "ASP",
        )
    )

    _assert_is_not_none(
        aspartate_group,
        "No aspartate anionic group was recognized.",
    )

    _assert_valid_charged_group(
        aspartate_group,
        expected_polarity=(
            "negative"
        ),
    )

    _assert_group_contains_atom_names(
        aspartate_group,
        [
            "OD1",
            "OD2",
        ],
    )

    normalized_type = normalize_text(
        aspartate_group.group_type,
        default="",
        lowercase=True,
    )

    _assert_true(
        (
            "carbox"
            in normalized_type
            or "aspart"
            in normalized_type
        ),
        "Aspartate should be classified as carboxylate-like.",
    )


def _test_recognize_glutamate_anion() -> None:
    """
    Test recognition of a glutamate carboxylate.
    """

    glutamate = _make_mock_glutamate(
        number=50,
        chain_id="B",
    )

    groups = (
        _call_recognize_anionic_groups(
            [
                glutamate
            ]
        )
    )

    glutamate_group = (
        _find_group_by_residue_name(
            groups,
            "GLU",
        )
    )

    _assert_is_not_none(
        glutamate_group,
        "No glutamate anionic group was recognized.",
    )

    _assert_valid_charged_group(
        glutamate_group,
        expected_polarity=(
            "negative"
        ),
    )

    _assert_group_contains_atom_names(
        glutamate_group,
        [
            "OE1",
            "OE2",
        ],
    )


def _test_recognize_multiple_protein_anions() -> None:
    """
    Test simultaneous recognition of aspartate and glutamate.
    """

    residues = [
        _make_mock_aspartate(
            number=40
        ),
        _make_mock_glutamate(
            number=50
        ),
    ]

    groups = (
        _call_recognize_anionic_groups(
            residues
        )
    )

    recognized_residue_names = {
        get_residue_name(
            group.residue
        )
        for group in groups
        if group.residue is not None
    }

    _assert_contains(
        recognized_residue_names,
        "ASP",
    )

    _assert_contains(
        recognized_residue_names,
        "GLU",
    )

    _assert_unique_group_ids(
        groups
    )


# =============================================================================
# 18.2.5. LIGAND CHARGED-GROUP RECOGNITION TESTS
# =============================================================================


def _test_recognize_ligand_cation() -> None:
    """
    Test recognition of a formally charged ligand nitrogen.
    """

    ligand = (
        _make_mock_cationic_ligand(
            center=(
                0.0,
                0.0,
                0.0,
            )
        )
    )

    groups = (
        _call_recognize_cationic_groups(
            [
                ligand
            ]
        )
    )

    _assert_not_empty(
        groups,
        "A formally charged ligand nitrogen should be recognized.",
    )

    ligand_groups = [
        group
        for group in groups
        if group.residue is ligand
    ]

    _assert_not_empty(
        ligand_groups,
        "No cationic group was associated with the ligand.",
    )

    cationic_group = (
        ligand_groups[0]
    )

    _assert_valid_charged_group(
        cationic_group,
        expected_polarity=(
            "positive"
        ),
    )

    _assert_group_contains_atom_names(
        cationic_group,
        [
            "N1"
        ],
    )


def _test_recognize_ligand_carboxylate() -> None:
    """
    Test recognition of a ligand carboxylate.
    """

    ligand = (
        _make_mock_carboxylate_ligand()
    )

    groups = (
        _call_recognize_anionic_groups(
            [
                ligand
            ]
        )
    )

    _assert_not_empty(
        groups,
        "Ligand carboxylate should be recognized.",
    )

    carboxylate_group = (
        _find_group_by_type_fragment(
            groups,
            "carbox",
        )
    )

    if carboxylate_group is None:
        carboxylate_group = (
            groups[0]
        )

    _assert_valid_charged_group(
        carboxylate_group,
        expected_polarity=(
            "negative"
        ),
    )

    _assert_group_contains_atom_names(
        carboxylate_group,
        [
            "O1",
            "O2",
        ],
    )


def _test_recognize_ligand_phosphate() -> None:
    """
    Test recognition of a ligand phosphate group.
    """

    ligand = (
        _make_mock_phosphate_ligand()
    )

    groups = (
        _call_recognize_anionic_groups(
            [
                ligand
            ]
        )
    )

    _assert_not_empty(
        groups,
        "Ligand phosphate should be recognized.",
    )

    phosphate_group = (
        _find_group_by_type_fragment(
            groups,
            "phosph",
        )
    )

    if phosphate_group is None:
        phosphate_group = (
            groups[0]
        )

    _assert_valid_charged_group(
        phosphate_group,
        expected_polarity=(
            "negative"
        ),
    )

    oxygen_count = sum(
        1
        for charged_atom
        in phosphate_group.atoms
        if charged_atom.element.upper()
        == "O"
    )

    _assert_true(
        oxygen_count >= 2,
        "A phosphate group should contain multiple oxygen atoms.",
    )


def _test_recognize_ligand_sulfonate() -> None:
    """
    Test recognition of a ligand sulfonate group.
    """

    ligand = (
        _make_mock_sulfonate_ligand()
    )

    groups = (
        _call_recognize_anionic_groups(
            [
                ligand
            ]
        )
    )

    _assert_not_empty(
        groups,
        "Ligand sulfonate should be recognized.",
    )

    sulfonate_group = (
        _find_group_by_type_fragment(
            groups,
            "sulfon",
        )
    )

    if sulfonate_group is None:
        sulfonate_group = (
            groups[0]
        )

    _assert_valid_charged_group(
        sulfonate_group,
        expected_polarity=(
            "negative"
        ),
    )


def _test_neutral_ligand_is_not_charged() -> None:
    """
    Test rejection of a neutral ligand.
    """

    neutral_ligand = (
        _make_mock_neutral_ligand()
    )

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            [
                neutral_ligand
            ]
        )
    )

    ligand_cations = [
        group
        for group in cationic_groups
        if group.residue
        is neutral_ligand
    ]

    ligand_anions = [
        group
        for group in anionic_groups
        if group.residue
        is neutral_ligand
    ]

    _assert_empty(
        ligand_cations,
        "Neutral ligand should not generate cationic groups.",
    )

    _assert_empty(
        ligand_anions,
        "Neutral ligand should not generate anionic groups.",
    )


# =============================================================================
# 18.2.6. RECOGNITION CONSISTENCY TESTS
# =============================================================================


def _test_recognition_polarity_separation() -> None:
    """
    Test correct separation between cationic and anionic groups.
    """

    residues = [
        _make_mock_lysine(
            number=10
        ),
        _make_mock_arginine(
            number=20
        ),
        _make_mock_aspartate(
            number=40
        ),
        _make_mock_glutamate(
            number=50
        ),
    ]

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            residues
        )
    )

    _assert_not_empty(
        cationic_groups
    )

    _assert_not_empty(
        anionic_groups
    )

    for group in cationic_groups:
        _assert_equal(
            group.polarity,
            "positive",
        )

        _assert_true(
            group.net_charge > 0.0
        )

    for group in anionic_groups:
        _assert_equal(
            group.polarity,
            "negative",
        )

        _assert_true(
            group.net_charge < 0.0
        )


def _test_recognition_does_not_duplicate_groups() -> None:
    """
    Test prevention of duplicate charged groups.
    """

    lysine = _make_mock_lysine(
        number=10
    )

    aspartate = (
        _make_mock_aspartate(
            number=40
        )
    )

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            [
                lysine,
                aspartate,
            ]
        )
    )

    _assert_unique_group_ids(
        cationic_groups
    )

    _assert_unique_group_ids(
        anionic_groups
    )

    cation_identities = [
        make_json_safe(
            charged_group_identity(
                group
            )
        )
        for group in cationic_groups
    ]

    anion_identities = [
        make_json_safe(
            charged_group_identity(
                group
            )
        )
        for group in anionic_groups
    ]

    _assert_equal(
        len(
            cation_identities
        ),
        len(
            {
                repr(identity)
                for identity
                in cation_identities
            }
        ),
        "Duplicate cationic-group identities were recognized.",
    )

    _assert_equal(
        len(
            anion_identities
        ),
        len(
            {
                repr(identity)
                for identity
                in anion_identities
            }
        ),
        "Duplicate anionic-group identities were recognized.",
    )


def _test_recognition_empty_source() -> None:
    """
    Test charged-group recognition with an empty source.
    """

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            []
        )
    )

    _assert_empty(
        cationic_groups
    )

    _assert_empty(
        anionic_groups
    )


# =============================================================================
# 18.2.7. GROUP-CENTER GEOMETRY TESTS
# =============================================================================


def _test_single_atom_group_center() -> None:
    """
    Test the center of a one-atom charged group.
    """

    group = _make_test_cation_group(
        center=(
            1.0,
            2.0,
            3.0,
        )
    )

    calculated_center = (
        _call_calculate_group_center(
            group
        )
    )

    expected_center = (
        group.atoms[0].coordinate
    )

    _assert_sequence_almost_equal(
        calculated_center,
        expected_center,
    )


def _test_two_atom_group_center() -> None:
    """
    Test the arithmetic center of a two-atom group.
    """

    group = _make_test_anion_group(
        center=(
            3.0,
            2.0,
            1.0,
        )
    )

    atom_coordinates = [
        charged_atom.coordinate
        for charged_atom in group.atoms
    ]

    expected_center = (
        sum(
            coordinate[0]
            for coordinate
            in atom_coordinates
        )
        / len(
            atom_coordinates
        ),
        sum(
            coordinate[1]
            for coordinate
            in atom_coordinates
        )
        / len(
            atom_coordinates
        ),
        sum(
            coordinate[2]
            for coordinate
            in atom_coordinates
        )
        / len(
            atom_coordinates
        ),
    )

    calculated_center = (
        _call_calculate_group_center(
            group
        )
    )

    _assert_sequence_almost_equal(
        calculated_center,
        expected_center,
    )


def _test_group_center_translation() -> None:
    """
    Test that group centers follow rigid translations.
    """

    group = _make_test_anion_group(
        center=(
            3.0,
            0.0,
            0.0,
        )
    )

    original_center = (
        _call_calculate_group_center(
            group
        )
    )

    translation = (
        4.0,
        -2.0,
        1.5,
    )

    for charged_atom in group.atoms:
        charged_atom.coordinate = (
            _translate_coordinate(
                charged_atom.coordinate,
                translation,
            )
        )

        if charged_atom.atom is not None:
            charged_atom.atom.coord = (
                charged_atom.coordinate
            )

    translated_center = (
        _call_calculate_group_center(
            group
        )
    )

    expected_center = (
        _translate_coordinate(
            original_center,
            translation,
        )
    )

    _assert_sequence_almost_equal(
        translated_center,
        expected_center,
    )


# =============================================================================
# 18.2.8. DISTANCE GEOMETRY TESTS
# =============================================================================


def _test_group_center_distance() -> None:
    """
    Test center-to-center charged-group distance.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            3.0,
            4.0,
            0.0,
        )
    )

    distance = (
        _call_calculate_group_distance(
            cation,
            anion,
        )
    )

    _assert_almost_equal(
        distance,
        5.0,
        tolerance=1e-6,
    )


def _test_geometry_center_distance() -> None:
    """
    Test center distance reported by SaltBridgeGeometry.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            3.0,
            0.0,
            0.0,
        )
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    _assert_valid_geometry(
        geometry
    )

    expected_distance = math.dist(
        cation.center,
        anion.center,
    )

    _assert_almost_equal(
        geometry.center_distance,
        expected_distance,
        tolerance=1e-6,
    )


def _test_geometry_atomic_distance_ordering() -> None:
    """
    Test ordering of minimum, mean, and maximum atom distances.
    """

    cation = _make_test_charged_group(
        group_id="cation_distance_order",
        group_type="guanidinium",
        polarity="positive",
        center=(
            0.0,
            0.0,
            0.0,
        ),
        net_charge=1.0,
        atom_names=(
            "N1",
            "N2",
            "N3",
        ),
    )

    anion = _make_test_anion_group(
        group_id="anion_distance_order",
        center=(
            3.0,
            0.0,
            0.0,
        ),
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    _assert_true(
        geometry.minimum_atom_distance
        <= geometry.mean_atom_distance
        <= geometry.maximum_atom_distance,
        (
            "Atomic distances must satisfy "
            "minimum <= mean <= maximum."
        ),
    )


def _test_geometry_closest_atom_pair() -> None:
    """
    Test resolution of the closest positive-negative atom pair.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            3.0,
            0.0,
            0.0,
        )
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    _assert_is_not_none(
        geometry.closest_positive_atom,
        "Closest positive atom should be resolved.",
    )

    _assert_is_not_none(
        geometry.closest_negative_atom,
        "Closest negative atom should be resolved.",
    )

    closest_distance = math.dist(
        get_atom_coordinate(
            geometry.closest_positive_atom
        ),
        get_atom_coordinate(
            geometry.closest_negative_atom
        ),
    )

    _assert_almost_equal(
        geometry.minimum_atom_distance,
        closest_distance,
        tolerance=1e-6,
    )


def _test_geometry_contact_count_nonnegative() -> None:
    """
    Test that atomic contact counts cannot be negative.
    """

    geometry = (
        _call_calculate_salt_bridge_geometry(
            _make_test_cation_group(),
            _make_test_anion_group(),
        )
    )

    _assert_true(
        geometry.contact_count >= 0
    )


# =============================================================================
# 18.2.9. VALID AND INVALID GEOMETRY TESTS
# =============================================================================


def _test_valid_short_range_geometry() -> None:
    """
    Test a geometrically valid short-range salt bridge.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            3.0,
            0.0,
            0.0,
        )
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    _assert_true(
        geometry.valid,
        (
            "A cation-anion pair separated by approximately "
            "3 Å should be geometrically valid."
        ),
    )

    _assert_is_none(
        geometry.rejection_reason,
        (
            "Valid geometry should not contain "
            "a rejection reason."
        ),
    )


def _test_invalid_long_range_geometry() -> None:
    """
    Test rejection of a distant charged-group pair.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            20.0,
            0.0,
            0.0,
        )
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    _assert_false(
        geometry.valid,
        "A 20 Å group separation must be rejected.",
    )

    _assert_is_not_none(
        geometry.rejection_reason,
        (
            "Rejected geometry should include "
            "a rejection reason."
        ),
    )


def _test_geometry_at_cutoff_boundary() -> None:
    """
    Test geometry near the configured distance cutoff.
    """

    config = resolve_config(
        None
    )

    cutoff_candidates = (
        "maximum_center_distance",
        "max_center_distance",
        "distance_cutoff",
        "maximum_distance",
        "max_distance",
    )

    cutoff = None

    for attribute_name in (
        cutoff_candidates
    ):
        candidate = get_value(
            config,
            attribute_name,
            None,
        )

        normalized_candidate = (
            safe_float(
                candidate
            )
        )

        if normalized_candidate is not None:
            cutoff = normalized_candidate
            break

    if cutoff is None:
        cutoff = 4.0

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            cutoff,
            0.0,
            0.0,
        )
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
            config,
        )
    )

    _assert_almost_equal(
        geometry.center_distance,
        cutoff,
        tolerance=1e-6,
    )

    _assert_is_instance(
        geometry.valid,
        bool,
    )


# =============================================================================
# 18.2.10. RIGID-TRANSFORMATION TESTS
# =============================================================================


def _translated_charged_group(
    group: ChargedGroup,
    translation: Sequence[float],
    *,
    group_id_suffix: str = "translated",
) -> ChargedGroup:
    """
    Create a translated copy of a charged group.
    """

    translated_atoms: List[
        ChargedAtom
    ] = []

    for atom_index, charged_atom in enumerate(
        group.atoms,
        start=1,
    ):
        translated_coordinate = (
            _translate_coordinate(
                charged_atom.coordinate,
                translation,
            )
        )

        translated_atoms.append(
            _make_test_charged_atom(
                name=charged_atom.name,
                element=(
                    charged_atom.element
                ),
                coordinate=(
                    translated_coordinate
                ),
                polarity=(
                    charged_atom.polarity
                ),
                effective_charge=(
                    charged_atom
                    .effective_charge
                ),
                source="self_test_translation",
                residue=group.residue,
                serial_number=(
                    atom_index
                ),
            )
        )

    translated_center = (
        _translate_coordinate(
            group.center,
            translation,
        )
    )

    return ChargedGroup(
        group_id=(
            f"{group.group_id}_"
            f"{group_id_suffix}"
        ),
        group_type=group.group_type,
        polarity=group.polarity,
        atoms=translated_atoms,
        residue=group.residue,
        center=translated_center,
        net_charge=group.net_charge,
        representative_atom=(
            translated_atoms[0].atom
            if translated_atoms
            else None
        ),
        source="self_test_translation",
        confidence=group.confidence,
        metadata={
            "self_test": True,
            "transformed": True,
        },
    )


def _rotated_charged_group_z(
    group: ChargedGroup,
    angle_degrees: float,
    *,
    origin: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    group_id_suffix: str = "rotated",
) -> ChargedGroup:
    """
    Create a z-axis-rotated copy of a charged group.
    """

    rotated_atoms: List[
        ChargedAtom
    ] = []

    for atom_index, charged_atom in enumerate(
        group.atoms,
        start=1,
    ):
        rotated_coordinate = (
            _rotate_coordinate_z(
                charged_atom.coordinate,
                angle_degrees,
                origin=origin,
            )
        )

        rotated_atoms.append(
            _make_test_charged_atom(
                name=charged_atom.name,
                element=(
                    charged_atom.element
                ),
                coordinate=(
                    rotated_coordinate
                ),
                polarity=(
                    charged_atom.polarity
                ),
                effective_charge=(
                    charged_atom
                    .effective_charge
                ),
                source="self_test_rotation",
                residue=group.residue,
                serial_number=(
                    atom_index
                ),
            )
        )

    rotated_center = (
        _rotate_coordinate_z(
            group.center,
            angle_degrees,
            origin=origin,
        )
    )

    return ChargedGroup(
        group_id=(
            f"{group.group_id}_"
            f"{group_id_suffix}"
        ),
        group_type=group.group_type,
        polarity=group.polarity,
        atoms=rotated_atoms,
        residue=group.residue,
        center=rotated_center,
        net_charge=group.net_charge,
        representative_atom=(
            rotated_atoms[0].atom
            if rotated_atoms
            else None
        ),
        source="self_test_rotation",
        confidence=group.confidence,
        metadata={
            "self_test": True,
            "transformed": True,
        },
    )


def _test_geometry_translation_invariance() -> None:
    """
    Test preservation of geometry under a common translation.
    """

    cation = _make_test_charged_group(
        group_id="translation_cation",
        group_type="guanidinium",
        polarity="positive",
        center=(
            0.0,
            0.0,
            0.0,
        ),
        net_charge=1.0,
        atom_names=(
            "N1",
            "N2",
            "N3",
        ),
    )

    anion = _make_test_anion_group(
        group_id="translation_anion",
        center=(
            3.2,
            0.4,
            0.0,
        ),
    )

    original_geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    translation = (
        10.0,
        -7.5,
        3.25,
    )

    translated_cation = (
        _translated_charged_group(
            cation,
            translation,
        )
    )

    translated_anion = (
        _translated_charged_group(
            anion,
            translation,
        )
    )

    translated_geometry = (
        _call_calculate_salt_bridge_geometry(
            translated_cation,
            translated_anion,
        )
    )

    _assert_almost_equal(
        translated_geometry.center_distance,
        original_geometry.center_distance,
    )

    _assert_almost_equal(
        translated_geometry.minimum_atom_distance,
        original_geometry.minimum_atom_distance,
    )

    _assert_almost_equal(
        translated_geometry.maximum_atom_distance,
        original_geometry.maximum_atom_distance,
    )

    _assert_almost_equal(
        translated_geometry.mean_atom_distance,
        original_geometry.mean_atom_distance,
    )

    _assert_equal(
        translated_geometry.valid,
        original_geometry.valid,
    )


def _test_geometry_rotation_invariance() -> None:
    """
    Test preservation of geometry under a common rigid rotation.
    """

    cation = _make_test_charged_group(
        group_id="rotation_cation",
        group_type="guanidinium",
        polarity="positive",
        center=(
            0.5,
            0.5,
            0.0,
        ),
        net_charge=1.0,
        atom_names=(
            "N1",
            "N2",
            "N3",
        ),
    )

    anion = _make_test_anion_group(
        group_id="rotation_anion",
        center=(
            3.3,
            1.1,
            0.0,
        ),
    )

    original_geometry = (
        _call_calculate_salt_bridge_geometry(
            cation,
            anion,
        )
    )

    rotated_cation = (
        _rotated_charged_group_z(
            cation,
            73.0,
        )
    )

    rotated_anion = (
        _rotated_charged_group_z(
            anion,
            73.0,
        )
    )

    rotated_geometry = (
        _call_calculate_salt_bridge_geometry(
            rotated_cation,
            rotated_anion,
        )
    )

    _assert_almost_equal(
        rotated_geometry.center_distance,
        original_geometry.center_distance,
        tolerance=1e-6,
    )

    _assert_almost_equal(
        rotated_geometry.minimum_atom_distance,
        original_geometry.minimum_atom_distance,
        tolerance=1e-6,
    )

    _assert_almost_equal(
        rotated_geometry.maximum_atom_distance,
        original_geometry.maximum_atom_distance,
        tolerance=1e-6,
    )

    _assert_almost_equal(
        rotated_geometry.mean_atom_distance,
        original_geometry.mean_atom_distance,
        tolerance=1e-6,
    )

    _assert_equal(
        rotated_geometry.valid,
        original_geometry.valid,
    )


# =============================================================================
# 18.2.11. RECOGNITION AND GEOMETRY COMBINED TESTS
# =============================================================================


def _test_recognized_groups_support_geometry() -> None:
    """
    Test geometry calculation using groups produced by recognition.
    """

    lysine = _make_mock_lysine(
        number=10,
        chain_id="A",
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
    )

    aspartate = (
        _make_mock_aspartate(
            number=40,
            chain_id="B",
            center=(
                3.0,
                0.0,
                0.0,
            ),
        )
    )

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            [
                lysine,
                aspartate,
            ]
        )
    )

    lysine_group = (
        _find_group_by_residue_name(
            cationic_groups,
            "LYS",
        )
    )

    aspartate_group = (
        _find_group_by_residue_name(
            anionic_groups,
            "ASP",
        )
    )

    _assert_is_not_none(
        lysine_group
    )

    _assert_is_not_none(
        aspartate_group
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            lysine_group,
            aspartate_group,
        )
    )

    _assert_valid_geometry(
        geometry
    )

    _assert_true(
        geometry.center_distance > 0.0
    )


def _test_recognition_geometry_far_negative_case() -> None:
    """
    Test a recognized charged pair that is geometrically too distant.
    """

    lysine = _make_mock_lysine(
        number=10,
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
    )

    glutamate = _make_mock_glutamate(
        number=50,
        center=(
            25.0,
            0.0,
            0.0,
        ),
    )

    cationic_groups, anionic_groups = (
        _call_recognize_charged_groups(
            [
                lysine,
                glutamate,
            ]
        )
    )

    lysine_group = (
        _find_group_by_residue_name(
            cationic_groups,
            "LYS",
        )
    )

    glutamate_group = (
        _find_group_by_residue_name(
            anionic_groups,
            "GLU",
        )
    )

    _assert_is_not_none(
        lysine_group
    )

    _assert_is_not_none(
        glutamate_group
    )

    geometry = (
        _call_calculate_salt_bridge_geometry(
            lysine_group,
            glutamate_group,
        )
    )

    _assert_false(
        geometry.valid
    )


# =============================================================================
# 18.2.12. SECTION TEST REGISTRY
# =============================================================================


def get_salt_bridge_recognition_geometry_tests(
) -> List[
    Tuple[
        str,
        Callable[
            [],
            Any,
        ],
    ]
]:
    """
    Return all Section 18.2 self-tests.

    Returns
    -------
    List[Tuple[str, Callable[[], Any]]]
        Named recognition and geometry tests.
    """

    return [
        (
            "18.2.recognition.lysine_cation",
            _test_recognize_lysine_cation,
        ),
        (
            "18.2.recognition.arginine_cation",
            _test_recognize_arginine_cation,
        ),
        (
            "18.2.recognition.protonated_histidine",
            _test_recognize_protonated_histidine_cation,
        ),
        (
            "18.2.recognition.multiple_protein_cations",
            _test_recognize_multiple_protein_cations,
        ),
        (
            "18.2.recognition.aspartate_anion",
            _test_recognize_aspartate_anion,
        ),
        (
            "18.2.recognition.glutamate_anion",
            _test_recognize_glutamate_anion,
        ),
        (
            "18.2.recognition.multiple_protein_anions",
            _test_recognize_multiple_protein_anions,
        ),
        (
            "18.2.recognition.ligand_cation",
            _test_recognize_ligand_cation,
        ),
        (
            "18.2.recognition.ligand_carboxylate",
            _test_recognize_ligand_carboxylate,
        ),
        (
            "18.2.recognition.ligand_phosphate",
            _test_recognize_ligand_phosphate,
        ),
        (
            "18.2.recognition.ligand_sulfonate",
            _test_recognize_ligand_sulfonate,
        ),
        (
            "18.2.recognition.neutral_ligand_negative",
            _test_neutral_ligand_is_not_charged,
        ),
        (
            "18.2.recognition.polarity_separation",
            _test_recognition_polarity_separation,
        ),
        (
            "18.2.recognition.no_duplicate_groups",
            _test_recognition_does_not_duplicate_groups,
        ),
        (
            "18.2.recognition.empty_source",
            _test_recognition_empty_source,
        ),
        (
            "18.2.geometry.single_atom_center",
            _test_single_atom_group_center,
        ),
        (
            "18.2.geometry.two_atom_center",
            _test_two_atom_group_center,
        ),
        (
            "18.2.geometry.center_translation",
            _test_group_center_translation,
        ),
        (
            "18.2.geometry.group_center_distance",
            _test_group_center_distance,
        ),
        (
            "18.2.geometry.reported_center_distance",
            _test_geometry_center_distance,
        ),
        (
            "18.2.geometry.atomic_distance_ordering",
            _test_geometry_atomic_distance_ordering,
        ),
        (
            "18.2.geometry.closest_atom_pair",
            _test_geometry_closest_atom_pair,
        ),
        (
            "18.2.geometry.contact_count",
            _test_geometry_contact_count_nonnegative,
        ),
        (
            "18.2.geometry.valid_short_range",
            _test_valid_short_range_geometry,
        ),
        (
            "18.2.geometry.invalid_long_range",
            _test_invalid_long_range_geometry,
        ),
        (
            "18.2.geometry.cutoff_boundary",
            _test_geometry_at_cutoff_boundary,
        ),
        (
            "18.2.geometry.translation_invariance",
            _test_geometry_translation_invariance,
        ),
        (
            "18.2.geometry.rotation_invariance",
            _test_geometry_rotation_invariance,
        ),
        (
            "18.2.integration.recognized_groups_geometry",
            _test_recognized_groups_support_geometry,
        ),
        (
            "18.2.integration.far_pair_negative",
            _test_recognition_geometry_far_negative_case,
        ),
    ]


# =============================================================================
# 18.2.13. SECTION RUNNER
# =============================================================================


def run_salt_bridge_recognition_geometry_tests(
    *,
    report: Optional[
        _SelfTestReport
    ] = None,
    raise_on_failure: bool = False,
    print_report: bool = False,
) -> _SelfTestReport:
    """
    Run all Section 18.2 recognition and geometry self-tests.

    Parameters
    ----------
    report
        Optional existing self-test report.
    raise_on_failure
        Whether the first failure should raise an exception.
    print_report
        Whether the resulting report should be printed.

    Returns
    -------
    _SelfTestReport
        Updated self-test report.
    """

    resolved_report = (
        _run_self_test_group(
            get_salt_bridge_recognition_geometry_tests(),
            report=report,
            raise_on_failure=(
                raise_on_failure
            ),
        )
    )

    if print_report:
        _print_self_test_report(
            resolved_report
        )

    return resolved_report



# =============================================================================
# 18.3. DETECTION AND CLASSIFICATION TESTS
# =============================================================================


# =============================================================================
# 18.3.1. DETECTION FUNCTION RESOLUTION
# =============================================================================


def _call_generate_salt_bridge_candidates(
    cationic_groups: Iterable[
        ChargedGroup
    ],
    anionic_groups: Iterable[
        ChargedGroup
    ],
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> List[Any]:
    """
    Call the available salt-bridge candidate-generation function.

    Parameters
    ----------
    cationic_groups
        Positive charged groups.
    anionic_groups
        Negative charged groups.
    config
        Optional salt-bridge configuration.

    Returns
    -------
    List[Any]
        Generated candidate pairs.
    """

    candidate_function_names = (
        "generate_salt_bridge_candidates",
        "build_salt_bridge_candidates",
        "find_salt_bridge_candidates",
        "enumerate_salt_bridge_candidates",
        "generate_candidate_pairs",
    )

    candidate_function = None

    for function_name in (
        candidate_function_names
    ):
        candidate = globals().get(
            function_name
        )

        if callable(
            candidate
        ):
            candidate_function = candidate
            break

    if candidate_function is None:
        return [
            (
                cation,
                anion,
            )
            for cation in cationic_groups
            for anion in anionic_groups
        ]

    try:
        candidates = candidate_function(
            cationic_groups,
            anionic_groups,
            config=config,
        )

    except TypeError:
        try:
            candidates = candidate_function(
                cationic_groups,
                anionic_groups,
                config,
            )

        except TypeError:
            candidates = candidate_function(
                cationic_groups,
                anionic_groups,
            )

    return list(
        candidates
        or []
    )


def _call_detect_salt_bridge_pair(
    cation: ChargedGroup,
    anion: ChargedGroup,
    config: Optional[
        SaltBridgeConfig
    ] = None,
    *,
    pose_id: Optional[
        Union[str, int]
    ] = None,
    model_id: Optional[
        Union[str, int]
    ] = None,
) -> Optional[
    SaltBridgeInteraction
]:
    """
    Call the available single-pair salt-bridge detector.

    Parameters
    ----------
    cation
        Positive charged group.
    anion
        Negative charged group.
    config
        Optional configuration.
    pose_id
        Optional pose identifier.
    model_id
        Optional model identifier.

    Returns
    -------
    Optional[SaltBridgeInteraction]
        Detected interaction or ``None``.
    """

    detection_function = (
        _resolve_self_test_callable(
            "detect_salt_bridge_pair",
            "evaluate_salt_bridge_pair",
            "detect_salt_bridge_between_groups",
            "build_salt_bridge_interaction",
            "analyze_salt_bridge_pair",
        )
    )

    keyword_variants = (
        {
            "config": config,
            "pose_id": pose_id,
            "model_id": model_id,
        },
        {
            "config": config,
        },
        {
            "pose_id": pose_id,
            "model_id": model_id,
        },
        {},
    )

    last_error = None

    for keyword_arguments in (
        keyword_variants
    ):
        filtered_arguments = {
            key: value
            for key, value
            in keyword_arguments.items()
            if value is not None
        }

        try:
            result = detection_function(
                cation,
                anion,
                **filtered_arguments,
            )

            if result is None:
                return None

            if isinstance(
                result,
                SaltBridgeInteraction,
            ):
                return result

            if isinstance(
                result,
                SaltBridgeGeometry,
            ):
                if not result.valid:
                    return None

                return _make_test_interaction(
                    cation=cation,
                    anion=anion,
                    geometry=result,
                    pose_id=pose_id,
                    model_id=model_id,
                )

            return result

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call the single-pair detector."
    ) from last_error


def _call_detect_salt_bridges_from_groups(
    cationic_groups: Iterable[
        ChargedGroup
    ],
    anionic_groups: Iterable[
        ChargedGroup
    ],
    config: Optional[
        SaltBridgeConfig
    ] = None,
    *,
    pose_id: Optional[
        Union[str, int]
    ] = None,
    model_id: Optional[
        Union[str, int]
    ] = None,
    include_invalid: bool = False,
) -> List[SaltBridgeInteraction]:
    """
    Call the available group-based salt-bridge detection function.

    Parameters
    ----------
    cationic_groups
        Positive charged groups.
    anionic_groups
        Negative charged groups.
    config
        Optional configuration.
    pose_id
        Optional pose identifier.
    model_id
        Optional model identifier.
    include_invalid
        Whether invalid interactions should be retained.

    Returns
    -------
    List[SaltBridgeInteraction]
        Detected interactions.
    """

    function_names = (
        "detect_salt_bridges_from_groups",
        "detect_salt_bridges",
        "find_salt_bridges",
        "analyze_salt_bridge_groups",
        "detect_group_salt_bridges",
    )

    detection_function = None

    for function_name in function_names:
        candidate = globals().get(
            function_name
        )

        if callable(
            candidate
        ):
            detection_function = candidate
            break

    if detection_function is None:
        detected_interactions: List[
            SaltBridgeInteraction
        ] = []

        for cation in cationic_groups:
            for anion in anionic_groups:
                interaction = (
                    _call_detect_salt_bridge_pair(
                        cation,
                        anion,
                        config,
                        pose_id=pose_id,
                        model_id=model_id,
                    )
                )

                if interaction is None:
                    continue

                if (
                    not include_invalid
                    and not interaction.geometry.valid
                ):
                    continue

                detected_interactions.append(
                    interaction
                )

        return detected_interactions

    keyword_variants = (
        {
            "config": config,
            "pose_id": pose_id,
            "model_id": model_id,
            "include_invalid": include_invalid,
        },
        {
            "config": config,
            "pose_id": pose_id,
            "model_id": model_id,
        },
        {
            "config": config,
            "include_invalid": include_invalid,
        },
        {
            "config": config,
        },
        {},
    )

    last_error = None

    for keyword_arguments in (
        keyword_variants
    ):
        filtered_arguments = {
            key: value
            for key, value
            in keyword_arguments.items()
            if value is not None
        }

        try:
            result = detection_function(
                cationic_groups,
                anionic_groups,
                **filtered_arguments,
            )

            if isinstance(
                result,
                SaltBridgeResult,
            ):
                return list(
                    result.interactions
                )

            return list(
                result
                or []
            )

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call the group-based detector."
    ) from last_error


def _call_analyze_salt_bridges(
    source: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
    *,
    pose_id: Optional[
        Union[str, int]
    ] = None,
    model_id: Optional[
        Union[str, int]
    ] = None,
) -> SaltBridgeResult:
    """
    Call the complete single-source salt-bridge analysis pipeline.

    Parameters
    ----------
    source
        Molecular source.
    config
        Optional salt-bridge configuration.
    pose_id
        Optional pose identifier.
    model_id
        Optional model identifier.

    Returns
    -------
    SaltBridgeResult
        Complete analysis result.
    """

    analysis_function = (
        _resolve_self_test_callable(
            "analyze_salt_bridges",
            "analyze_salt_bridges_with_statistics",
            "detect_salt_bridges_in_source",
            "run_salt_bridge_analysis",
        )
    )

    keyword_variants = (
        {
            "config": config,
            "pose_id": pose_id,
            "model_id": model_id,
        },
        {
            "config": config,
        },
        {
            "pose_id": pose_id,
            "model_id": model_id,
        },
        {},
    )

    last_error = None

    for keyword_arguments in (
        keyword_variants
    ):
        filtered_arguments = {
            key: value
            for key, value
            in keyword_arguments.items()
            if value is not None
        }

        try:
            result = analysis_function(
                source,
                **filtered_arguments,
            )

            if isinstance(
                result,
                SaltBridgeResult,
            ):
                return result

            if isinstance(
                result,
                Iterable,
            ):
                interactions = list(
                    result
                )

                return _make_test_result(
                    interactions=interactions,
                    pose_id=pose_id,
                    model_id=model_id,
                )

            raise SaltBridgeSelfTestError(
                "Complete analysis did not return "
                "SaltBridgeResult-compatible data."
            )

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call the complete salt-bridge analysis pipeline."
    ) from last_error


# =============================================================================
# 18.3.2. CLASSIFICATION AND SCORING FUNCTION RESOLUTION
# =============================================================================


def _call_classify_salt_bridge_strength(
    geometry_or_interaction: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> str:
    """
    Call the available salt-bridge strength classifier.

    Parameters
    ----------
    geometry_or_interaction
        SaltBridgeGeometry or SaltBridgeInteraction.
    config
        Optional configuration.

    Returns
    -------
    str
        Strength category.
    """

    classification_function = (
        _resolve_self_test_callable(
            "classify_salt_bridge_strength",
            "classify_salt_bridge",
            "classify_interaction_strength",
            "assign_salt_bridge_strength",
        )
    )

    try:
        strength = classification_function(
            geometry_or_interaction,
            config=config,
        )

    except TypeError:
        try:
            strength = classification_function(
                geometry_or_interaction,
                config,
            )

        except TypeError:
            strength = classification_function(
                geometry_or_interaction
            )

    if isinstance(
        strength,
        Mapping,
    ):
        strength = (
            strength.get(
                "strength"
            )
            or strength.get(
                "classification"
            )
            or strength.get(
                "label"
            )
        )

    normalized_strength = normalize_text(
        strength,
        default="",
        lowercase=True,
    )

    if not normalized_strength:
        raise SaltBridgeSelfTestError(
            "Strength classification returned an empty value."
        )

    return normalized_strength


def _call_score_salt_bridge(
    interaction_or_geometry: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
    *,
    cation: Optional[
        ChargedGroup
    ] = None,
    anion: Optional[
        ChargedGroup
    ] = None,
) -> float:
    """
    Call the available salt-bridge scoring function.

    Parameters
    ----------
    interaction_or_geometry
        Interaction or geometry to score.
    config
        Optional configuration.
    cation
        Optional cation group.
    anion
        Optional anion group.

    Returns
    -------
    float
        Salt-bridge score.
    """

    scoring_function = (
        _resolve_self_test_callable(
            "score_salt_bridge",
            "calculate_salt_bridge_score",
            "score_salt_bridge_interaction",
            "calculate_interaction_score",
        )
    )

    call_variants = [
        (
            (
                interaction_or_geometry,
            ),
            {
                "config": config,
                "cation": cation,
                "anion": anion,
            },
        ),
        (
            (
                interaction_or_geometry,
                cation,
                anion,
            ),
            {
                "config": config,
            },
        ),
        (
            (
                interaction_or_geometry,
            ),
            {
                "config": config,
            },
        ),
        (
            (
                interaction_or_geometry,
            ),
            {},
        ),
    ]

    last_error = None

    for positional_arguments, keyword_arguments in (
        call_variants
    ):
        filtered_positional_arguments = tuple(
            argument
            for argument in positional_arguments
            if argument is not None
        )

        filtered_keyword_arguments = {
            key: value
            for key, value
            in keyword_arguments.items()
            if value is not None
        }

        try:
            score = scoring_function(
                *filtered_positional_arguments,
                **filtered_keyword_arguments,
            )

            if isinstance(
                score,
                Mapping,
            ):
                score = (
                    score.get(
                        "score"
                    )
                    or score.get(
                        "total_score"
                    )
                )

            normalized_score = safe_float(
                score
            )

            if normalized_score is None:
                raise SaltBridgeSelfTestError(
                    "Scoring function returned "
                    "a non-numeric value."
                )

            return normalized_score

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call the salt-bridge scoring function."
    ) from last_error


# =============================================================================
# 18.3.3. CANDIDATE-GENERATION TESTS
# =============================================================================


def _test_candidate_generation_cartesian_product() -> None:
    """
    Test candidate generation for all cation-anion combinations.
    """

    cations = [
        _make_test_cation_group(
            group_id="cation_1",
        ),
        _make_test_cation_group(
            group_id="cation_2",
            center=(
                1.0,
                0.0,
                0.0,
            ),
        ),
    ]

    anions = [
        _make_test_anion_group(
            group_id="anion_1",
        ),
        _make_test_anion_group(
            group_id="anion_2",
            center=(
                4.0,
                0.0,
                0.0,
            ),
        ),
        _make_test_anion_group(
            group_id="anion_3",
            center=(
                5.0,
                0.0,
                0.0,
            ),
        ),
    ]

    candidates = (
        _call_generate_salt_bridge_candidates(
            cations,
            anions,
        )
    )

    _assert_length(
        candidates,
        6,
        (
            "Two cations and three anions should "
            "generate six candidate pairs."
        ),
    )


def _test_candidate_generation_empty_cations() -> None:
    """
    Test candidate generation with no cations.
    """

    candidates = (
        _call_generate_salt_bridge_candidates(
            [],
            [
                _make_test_anion_group()
            ],
        )
    )

    _assert_empty(
        candidates
    )


def _test_candidate_generation_empty_anions() -> None:
    """
    Test candidate generation with no anions.
    """

    candidates = (
        _call_generate_salt_bridge_candidates(
            [
                _make_test_cation_group()
            ],
            [],
        )
    )

    _assert_empty(
        candidates
    )


def _test_candidate_generation_polarity_order() -> None:
    """
    Test that candidate pairs preserve cation-anion ordering.
    """

    cation = _make_test_cation_group(
        group_id="ordered_cation"
    )

    anion = _make_test_anion_group(
        group_id="ordered_anion"
    )

    candidates = (
        _call_generate_salt_bridge_candidates(
            [
                cation
            ],
            [
                anion
            ],
        )
    )

    _assert_length(
        candidates,
        1,
    )

    candidate = candidates[0]

    if isinstance(
        candidate,
        Mapping,
    ):
        candidate_cation = (
            candidate.get(
                "cation"
            )
            or candidate.get(
                "positive_group"
            )
        )

        candidate_anion = (
            candidate.get(
                "anion"
            )
            or candidate.get(
                "negative_group"
            )
        )

    else:
        candidate_cation = candidate[0]
        candidate_anion = candidate[1]

    _assert_equal(
        candidate_cation.polarity,
        "positive",
    )

    _assert_equal(
        candidate_anion.polarity,
        "negative",
    )


# =============================================================================
# 18.3.4. SINGLE-PAIR DETECTION TESTS
# =============================================================================


def _test_detect_valid_salt_bridge_pair() -> None:
    """
    Test detection of a valid close cation-anion pair.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            3.0,
            0.0,
            0.0,
        )
    )

    interaction = (
        _call_detect_salt_bridge_pair(
            cation,
            anion,
            pose_id=1,
            model_id="model_1",
        )
    )

    _assert_is_not_none(
        interaction,
        "A valid cation-anion pair should be detected.",
    )

    _assert_valid_interaction(
        interaction,
        expected_valid=True,
    )

    _assert_equal(
        interaction.cation.polarity,
        "positive",
    )

    _assert_equal(
        interaction.anion.polarity,
        "negative",
    )


def _test_reject_distant_salt_bridge_pair() -> None:
    """
    Test rejection of a distant cation-anion pair.
    """

    cation = _make_test_cation_group(
        center=(
            0.0,
            0.0,
            0.0,
        )
    )

    anion = _make_test_anion_group(
        center=(
            20.0,
            0.0,
            0.0,
        )
    )

    interaction = (
        _call_detect_salt_bridge_pair(
            cation,
            anion,
        )
    )

    if interaction is None:
        return

    _assert_false(
        interaction.geometry.valid,
        "A 20 Å pair must not be accepted as valid.",
    )


def _test_detect_pair_preserves_identifiers() -> None:
    """
    Test preservation of pose and model identifiers.
    """

    interaction = (
        _call_detect_salt_bridge_pair(
            _make_test_cation_group(),
            _make_test_anion_group(),
            pose_id=7,
            model_id="pose_model_7",
        )
    )

    _assert_is_not_none(
        interaction
    )

    if interaction.pose_id is not None:
        _assert_equal(
            interaction.pose_id,
            7,
        )

    if interaction.model_id is not None:
        _assert_equal(
            interaction.model_id,
            "pose_model_7",
        )


def _test_detect_pair_builds_unique_identifier() -> None:
    """
    Test construction of a non-empty interaction identifier.
    """

    interaction = (
        _call_detect_salt_bridge_pair(
            _make_test_cation_group(
                group_id="identifier_cation"
            ),
            _make_test_anion_group(
                group_id="identifier_anion"
            ),
        )
    )

    _assert_is_not_none(
        interaction
    )

    _assert_true(
        bool(
            str(
                interaction.interaction_id
            ).strip()
        ),
        "Detected interaction must have an identifier.",
    )


# =============================================================================
# 18.3.5. MULTIPLE-GROUP DETECTION TESTS
# =============================================================================


def _test_detect_multiple_valid_pairs() -> None:
    """
    Test detection across multiple charged groups.
    """

    cations = [
        _make_test_cation_group(
            group_id="multi_cation_1",
            center=(
                0.0,
                0.0,
                0.0,
            ),
        ),
        _make_test_cation_group(
            group_id="multi_cation_2",
            center=(
                0.0,
                5.0,
                0.0,
            ),
        ),
    ]

    anions = [
        _make_test_anion_group(
            group_id="multi_anion_1",
            center=(
                3.0,
                0.0,
                0.0,
            ),
        ),
        _make_test_anion_group(
            group_id="multi_anion_2",
            center=(
                3.0,
                5.0,
                0.0,
            ),
        ),
    ]

    interactions = (
        _call_detect_salt_bridges_from_groups(
            cations,
            anions,
        )
    )

    valid_interactions = [
        interaction
        for interaction in interactions
        if interaction.geometry.valid
    ]

    _assert_true(
        len(
            valid_interactions
        ) >= 2,
        (
            "The two close cation-anion pairs "
            "should be detected."
        ),
    )


def _test_detection_excludes_invalid_by_default() -> None:
    """
    Test default exclusion of distant invalid candidates.
    """

    interactions = (
        _call_detect_salt_bridges_from_groups(
            [
                _make_test_cation_group()
            ],
            [
                _make_test_anion_group(
                    center=(
                        20.0,
                        0.0,
                        0.0,
                    )
                )
            ],
            include_invalid=False,
        )
    )

    _assert_true(
        all(
            interaction.geometry.valid
            for interaction
            in interactions
        ),
        (
            "Default detection output should not "
            "contain invalid interactions."
        ),
    )


def _test_detection_no_duplicate_interactions() -> None:
    """
    Test that one group pair does not create duplicate interactions.
    """

    cation = _make_test_cation_group(
        group_id="duplicate_test_cation"
    )

    anion = _make_test_anion_group(
        group_id="duplicate_test_anion"
    )

    interactions = (
        _call_detect_salt_bridges_from_groups(
            [
                cation
            ],
            [
                anion
            ],
        )
    )

    interaction_keys = [
        (
            interaction.cation.group_id,
            interaction.anion.group_id,
            interaction.pose_id,
            interaction.model_id,
        )
        for interaction in interactions
        if interaction.geometry.valid
    ]

    _assert_equal(
        len(
            interaction_keys
        ),
        len(
            set(
                interaction_keys
            )
        ),
        "Detection produced duplicate interactions.",
    )


# =============================================================================
# 18.3.6. STRENGTH CLASSIFICATION TESTS
# =============================================================================


def _test_classify_strong_salt_bridge() -> None:
    """
    Test classification of a short strong salt bridge.
    """

    geometry = _make_test_geometry(
        center_distance=2.8,
        minimum_atom_distance=2.5,
        maximum_atom_distance=3.1,
        mean_atom_distance=2.8,
        contact_count=3,
        valid=True,
    )

    strength = (
        _call_classify_salt_bridge_strength(
            geometry
        )
    )

    _assert_equal(
        strength,
        STRENGTH_STRONG,
        (
            "A short, multi-contact interaction "
            "should be classified as strong."
        ),
    )


def _test_classify_moderate_salt_bridge() -> None:
    """
    Test classification of an intermediate-distance salt bridge.
    """

    geometry = _make_test_geometry(
        center_distance=3.6,
        minimum_atom_distance=3.2,
        maximum_atom_distance=4.0,
        mean_atom_distance=3.6,
        contact_count=1,
        valid=True,
    )

    strength = (
        _call_classify_salt_bridge_strength(
            geometry
        )
    )

    _assert_contains(
        {
            STRENGTH_MODERATE,
            STRENGTH_WEAK,
        },
        strength,
        (
            "An intermediate geometry should be "
            "classified as moderate or weak."
        ),
    )


def _test_classify_weak_salt_bridge() -> None:
    """
    Test classification of a near-cutoff salt bridge.
    """

    geometry = _make_test_geometry(
        center_distance=4.5,
        minimum_atom_distance=4.0,
        maximum_atom_distance=5.0,
        mean_atom_distance=4.5,
        contact_count=1,
        valid=True,
    )

    strength = (
        _call_classify_salt_bridge_strength(
            geometry
        )
    )

    _assert_contains(
        {
            STRENGTH_WEAK,
            STRENGTH_MODERATE,
        },
        strength,
        (
            "A near-cutoff valid interaction should "
            "not be classified as strong."
        ),
    )

    _assert_not_equal(
        strength,
        STRENGTH_STRONG,
    )


def _test_classify_invalid_as_rejected() -> None:
    """
    Test classification of invalid geometry.
    """

    geometry = _make_test_geometry(
        center_distance=10.0,
        minimum_atom_distance=9.5,
        maximum_atom_distance=10.5,
        mean_atom_distance=10.0,
        contact_count=0,
        valid=False,
        rejection_reason="distance_cutoff",
    )

    strength = (
        _call_classify_salt_bridge_strength(
            geometry
        )
    )

    _assert_equal(
        strength,
        STRENGTH_REJECTED,
        "Invalid geometry should be classified as rejected.",
    )


def _test_strength_improves_with_shorter_distance() -> None:
    """
    Test monotonic strength behavior with decreasing distance.
    """

    strong_geometry = _make_test_geometry(
        center_distance=2.8,
        minimum_atom_distance=2.5,
        maximum_atom_distance=3.1,
        mean_atom_distance=2.8,
        contact_count=3,
        valid=True,
    )

    weaker_geometry = _make_test_geometry(
        center_distance=4.3,
        minimum_atom_distance=3.9,
        maximum_atom_distance=4.7,
        mean_atom_distance=4.3,
        contact_count=1,
        valid=True,
    )

    strong_class = (
        _call_classify_salt_bridge_strength(
            strong_geometry
        )
    )

    weaker_class = (
        _call_classify_salt_bridge_strength(
            weaker_geometry
        )
    )

    strength_rank = {
        STRENGTH_REJECTED: 0,
        STRENGTH_WEAK: 1,
        STRENGTH_MODERATE: 2,
        STRENGTH_STRONG: 3,
    }

    _assert_true(
        strength_rank.get(
            strong_class,
            -1,
        )
        >= strength_rank.get(
            weaker_class,
            -1,
        ),
        (
            "Shorter geometry should not receive "
            "a weaker classification."
        ),
    )


# =============================================================================
# 18.3.7. SCORING TESTS
# =============================================================================


def _test_valid_interaction_has_positive_score() -> None:
    """
    Test positive scoring of a valid salt bridge.
    """

    interaction = _make_test_interaction(
        geometry=_make_test_geometry(
            center_distance=3.0,
            minimum_atom_distance=2.7,
            maximum_atom_distance=3.3,
            mean_atom_distance=3.0,
            contact_count=2,
            valid=True,
        ),
    )

    score = _call_score_salt_bridge(
        interaction
    )

    _assert_true(
        score > 0.0,
        "A valid salt bridge should have a positive score.",
    )


def _test_invalid_interaction_has_zero_or_minimal_score() -> None:
    """
    Test scoring of an invalid salt bridge.
    """

    interaction = _make_test_interaction(
        geometry=_make_test_geometry(
            center_distance=12.0,
            minimum_atom_distance=11.5,
            maximum_atom_distance=12.5,
            mean_atom_distance=12.0,
            contact_count=0,
            valid=False,
            rejection_reason="distance_cutoff",
        ),
        strength=STRENGTH_REJECTED,
        score=0.0,
    )

    score = _call_score_salt_bridge(
        interaction
    )

    _assert_true(
        score <= 0.0
        or math.isclose(
            score,
            0.0,
            abs_tol=1e-6,
        ),
        "Rejected salt bridges should not receive a positive score.",
    )


def _test_shorter_interaction_scores_higher() -> None:
    """
    Test that shorter valid interactions score at least as highly.
    """

    short_interaction = (
        _make_test_interaction(
            interaction_id="short_score_test",
            geometry=_make_test_geometry(
                center_distance=2.8,
                minimum_atom_distance=2.5,
                maximum_atom_distance=3.1,
                mean_atom_distance=2.8,
                contact_count=3,
                valid=True,
            ),
        )
    )

    long_interaction = (
        _make_test_interaction(
            interaction_id="long_score_test",
            geometry=_make_test_geometry(
                center_distance=4.4,
                minimum_atom_distance=4.0,
                maximum_atom_distance=4.8,
                mean_atom_distance=4.4,
                contact_count=1,
                valid=True,
            ),
        )
    )

    short_score = (
        _call_score_salt_bridge(
            short_interaction
        )
    )

    long_score = (
        _call_score_salt_bridge(
            long_interaction
        )
    )

    _assert_true(
        short_score >= long_score,
        (
            "A shorter valid interaction should not "
            "score below a longer one."
        ),
    )


def _test_multiple_contacts_do_not_reduce_score() -> None:
    """
    Test score behavior with additional atomic contacts.
    """

    one_contact = _make_test_interaction(
        interaction_id="one_contact",
        geometry=_make_test_geometry(
            center_distance=3.2,
            minimum_atom_distance=2.9,
            maximum_atom_distance=3.5,
            mean_atom_distance=3.2,
            contact_count=1,
            valid=True,
        ),
    )

    three_contacts = _make_test_interaction(
        interaction_id="three_contacts",
        geometry=_make_test_geometry(
            center_distance=3.2,
            minimum_atom_distance=2.9,
            maximum_atom_distance=3.5,
            mean_atom_distance=3.2,
            contact_count=3,
            valid=True,
        ),
    )

    one_contact_score = (
        _call_score_salt_bridge(
            one_contact
        )
    )

    three_contact_score = (
        _call_score_salt_bridge(
            three_contacts
        )
    )

    _assert_true(
        three_contact_score
        >= one_contact_score,
        (
            "Additional atomic contacts should not "
            "reduce the interaction score."
        ),
    )


def _test_stronger_charge_does_not_reduce_score() -> None:
    """
    Test score behavior with increased charge magnitude.
    """

    weak_cation = _make_test_cation_group(
        group_id="weak_charge_cation",
        net_charge=0.5,
    )

    strong_cation = _make_test_cation_group(
        group_id="strong_charge_cation",
        net_charge=1.0,
    )

    anion = _make_test_anion_group(
        net_charge=-1.0,
    )

    geometry = _make_test_geometry(
        center_distance=3.0,
        minimum_atom_distance=2.7,
        maximum_atom_distance=3.3,
        mean_atom_distance=3.0,
        contact_count=2,
        valid=True,
    )

    weak_interaction = _make_test_interaction(
        interaction_id="weak_charge",
        cation=weak_cation,
        anion=anion,
        geometry=geometry,
    )

    strong_interaction = _make_test_interaction(
        interaction_id="strong_charge",
        cation=strong_cation,
        anion=anion,
        geometry=geometry,
    )

    weak_score = _call_score_salt_bridge(
        weak_interaction,
        cation=weak_cation,
        anion=anion,
    )

    strong_score = _call_score_salt_bridge(
        strong_interaction,
        cation=strong_cation,
        anion=anion,
    )

    _assert_true(
        strong_score >= weak_score,
        (
            "Increasing charge magnitude should not "
            "reduce the salt-bridge score."
        ),
    )


def _test_score_is_finite() -> None:
    """
    Test that salt-bridge scores are finite.
    """

    interaction = _make_test_interaction()

    score = _call_score_salt_bridge(
        interaction
    )

    _assert_true(
        math.isfinite(
            score
        ),
        "Salt-bridge score must be finite.",
    )


# =============================================================================
# 18.3.8. DETECTION-CLASSIFICATION CONSISTENCY TESTS
# =============================================================================


def _test_detected_interaction_strength_matches_classifier() -> None:
    """
    Test consistency between detection and standalone classification.
    """

    interaction = (
        _call_detect_salt_bridge_pair(
            _make_test_cation_group(),
            _make_test_anion_group(),
        )
    )

    _assert_is_not_none(
        interaction
    )

    classified_strength = (
        _call_classify_salt_bridge_strength(
            interaction.geometry
        )
    )

    _assert_equal(
        normalize_text(
            interaction.strength,
            default="",
            lowercase=True,
        ),
        classified_strength,
        (
            "Detected interaction strength should match "
            "the standalone classifier."
        ),
    )


def _test_detected_interaction_score_matches_scorer() -> None:
    """
    Test consistency between detection and standalone scoring.
    """

    interaction = (
        _call_detect_salt_bridge_pair(
            _make_test_cation_group(),
            _make_test_anion_group(),
        )
    )

    _assert_is_not_none(
        interaction
    )

    calculated_score = (
        _call_score_salt_bridge(
            interaction
        )
    )

    _assert_almost_equal(
        interaction.score,
        calculated_score,
        tolerance=1e-6,
        message=(
            "Detected interaction score should match "
            "the standalone scoring function."
        ),
    )


def _test_detected_valid_interaction_not_rejected() -> None:
    """
    Test that valid detected interactions are not marked rejected.
    """

    interaction = (
        _call_detect_salt_bridge_pair(
            _make_test_cation_group(),
            _make_test_anion_group(),
        )
    )

    _assert_is_not_none(
        interaction
    )

    _assert_true(
        interaction.geometry.valid
    )

    _assert_not_equal(
        normalize_text(
            interaction.strength,
            default="",
            lowercase=True,
        ),
        STRENGTH_REJECTED,
    )

    _assert_true(
        interaction.score > 0.0
    )


# =============================================================================
# 18.3.9. COMPLETE PIPELINE TESTS
# =============================================================================


def _test_complete_pipeline_positive_case() -> None:
    """
    Test complete recognition-to-detection analysis.
    """

    structure = _make_test_structure(
        model_id=1
    )

    lysine = _make_mock_lysine(
        number=10,
        chain_id="A",
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
        structure=structure,
    )

    aspartate = (
        _make_mock_aspartate(
            number=40,
            chain_id="B",
            center=(
                3.0,
                0.0,
                0.0,
            ),
            structure=structure,
        )
    )

    result = _call_analyze_salt_bridges(
        [
            lysine,
            aspartate,
        ],
        pose_id=1,
        model_id="model_1",
    )

    _assert_valid_result(
        result
    )

    _assert_not_empty(
        result.cationic_groups,
        "Pipeline should recognize cationic groups.",
    )

    _assert_not_empty(
        result.anionic_groups,
        "Pipeline should recognize anionic groups.",
    )

    valid_interactions = [
        interaction
        for interaction in result.interactions
        if interaction.geometry.valid
    ]

    _assert_not_empty(
        valid_interactions,
        (
            "Pipeline should detect at least one "
            "valid salt bridge."
        ),
    )


def _test_complete_pipeline_distant_negative_case() -> None:
    """
    Test complete pipeline with charged groups too far apart.
    """

    lysine = _make_mock_lysine(
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        )
    )

    aspartate = (
        _make_mock_aspartate(
            center=(
                25.0,
                0.0,
                0.0,
            )
        )
    )

    result = _call_analyze_salt_bridges(
        [
            lysine,
            aspartate,
        ]
    )

    valid_interactions = [
        interaction
        for interaction in result.interactions
        if interaction.geometry.valid
    ]

    _assert_empty(
        valid_interactions,
        (
            "Distant charged groups should not produce "
            "valid salt bridges."
        ),
    )


def _test_complete_pipeline_neutral_negative_case() -> None:
    """
    Test complete pipeline with only neutral residues.
    """

    neutral_ligand = (
        _make_mock_neutral_ligand()
    )

    result = _call_analyze_salt_bridges(
        [
            neutral_ligand
        ]
    )

    valid_interactions = [
        interaction
        for interaction in result.interactions
        if interaction.geometry.valid
    ]

    _assert_empty(
        valid_interactions
    )


def _test_complete_pipeline_mixed_pairs() -> None:
    """
    Test complete pipeline with close and distant charge pairs.
    """

    lysine_close = _make_mock_lysine(
        number=10,
        chain_id="A",
        nz_coordinate=(
            0.0,
            0.0,
            0.0,
        ),
    )

    arginine_far = _make_mock_arginine(
        number=20,
        chain_id="A",
        center=(
            30.0,
            0.0,
            0.0,
        ),
    )

    aspartate_close = (
        _make_mock_aspartate(
            number=40,
            chain_id="B",
            center=(
                3.0,
                0.0,
                0.0,
            ),
        )
    )

    result = _call_analyze_salt_bridges(
        [
            lysine_close,
            arginine_far,
            aspartate_close,
        ]
    )

    valid_interactions = [
        interaction
        for interaction in result.interactions
        if interaction.geometry.valid
    ]

    _assert_not_empty(
        valid_interactions
    )

    for interaction in valid_interactions:
        _assert_true(
            interaction.geometry.center_distance
            < 20.0,
            (
                "Distant candidate should not appear "
                "as a valid interaction."
            ),
        )


def _test_complete_pipeline_result_identifiers() -> None:
    """
    Test propagation of pose and model identifiers.
    """

    result = _call_analyze_salt_bridges(
        [
            _make_mock_lysine(),
            _make_mock_aspartate(),
        ],
        pose_id=8,
        model_id="model_8",
    )

    if result.pose_id is not None:
        _assert_equal(
            result.pose_id,
            8,
        )

    if result.model_id is not None:
        _assert_equal(
            result.model_id,
            "model_8",
        )

    for interaction in result.interactions:
        if interaction.pose_id is not None:
            _assert_equal(
                interaction.pose_id,
                8,
            )

        if interaction.model_id is not None:
            _assert_equal(
                interaction.model_id,
                "model_8",
            )


# =============================================================================
# 18.3.10. RESULT-INVARIANT TESTS
# =============================================================================


def _test_all_detected_interactions_have_valid_polarities() -> None:
    """
    Test cation-anion polarity invariants in detection output.
    """

    result = _call_analyze_salt_bridges(
        [
            _make_mock_lysine(),
            _make_mock_arginine(
                center=(
                    0.0,
                    5.0,
                    0.0,
                )
            ),
            _make_mock_aspartate(),
            _make_mock_glutamate(
                center=(
                    3.0,
                    5.0,
                    0.0,
                )
            ),
        ]
    )

    for interaction in result.interactions:
        _assert_equal(
            interaction.cation.polarity,
            "positive",
        )

        _assert_equal(
            interaction.anion.polarity,
            "negative",
        )

        _assert_true(
            interaction.cation.net_charge
            > 0.0
        )

        _assert_true(
            interaction.anion.net_charge
            < 0.0
        )


def _test_all_detected_scores_are_finite() -> None:
    """
    Test that all detected interaction scores are finite.
    """

    result = _call_analyze_salt_bridges(
        [
            _make_mock_lysine(),
            _make_mock_aspartate(),
        ]
    )

    for interaction in result.interactions:
        _assert_true(
            math.isfinite(
                interaction.score
            ),
            (
                "Detected interaction score "
                "must be finite."
            ),
        )


def _test_all_detected_geometries_are_consistent() -> None:
    """
    Test geometry invariants in all detected interactions.
    """

    result = _call_analyze_salt_bridges(
        [
            _make_mock_lysine(),
            _make_mock_arginine(
                center=(
                    0.0,
                    4.0,
                    0.0,
                )
            ),
            _make_mock_aspartate(),
            _make_mock_glutamate(
                center=(
                    3.0,
                    4.0,
                    0.0,
                )
            ),
        ]
    )

    for interaction in result.interactions:
        geometry = interaction.geometry

        _assert_true(
            geometry.minimum_atom_distance
            <= geometry.mean_atom_distance
            <= geometry.maximum_atom_distance
        )

        _assert_true(
            geometry.center_distance
            >= 0.0
        )

        _assert_true(
            geometry.contact_count
            >= 0
        )


def _test_valid_interactions_have_positive_scores() -> None:
    """
    Test that all valid pipeline interactions have positive scores.
    """

    result = _call_analyze_salt_bridges(
        [
            _make_mock_lysine(),
            _make_mock_aspartate(),
        ]
    )

    for interaction in result.interactions:
        if not interaction.geometry.valid:
            continue

        _assert_true(
            interaction.score > 0.0,
            (
                "Valid interactions should have "
                "positive scores."
            ),
        )


def _test_rejected_interactions_are_not_strong() -> None:
    """
    Test that rejected interactions cannot be strong.
    """

    rejected_interaction = (
        _make_test_interaction(
            geometry=_make_test_geometry(
                center_distance=15.0,
                minimum_atom_distance=14.5,
                maximum_atom_distance=15.5,
                mean_atom_distance=15.0,
                contact_count=0,
                valid=False,
                rejection_reason=(
                    "distance_cutoff"
                ),
            ),
            strength=STRENGTH_REJECTED,
            score=0.0,
        )
    )

    strength = (
        _call_classify_salt_bridge_strength(
            rejected_interaction.geometry
        )
    )

    _assert_not_equal(
        strength,
        STRENGTH_STRONG
    )

    _assert_equal(
        strength,
        STRENGTH_REJECTED
    )


# =============================================================================
# 18.3.11. SECTION TEST REGISTRY
# =============================================================================


def get_salt_bridge_detection_classification_tests(
) -> List[
    Tuple[
        str,
        Callable[
            [],
            Any,
        ],
    ]
]:
    """
    Return all Section 18.3 self-tests.

    Returns
    -------
    List[Tuple[str, Callable[[], Any]]]
        Named detection and classification tests.
    """

    return [
        (
            "18.3.candidates.cartesian_product",
            _test_candidate_generation_cartesian_product,
        ),
        (
            "18.3.candidates.empty_cations",
            _test_candidate_generation_empty_cations,
        ),
        (
            "18.3.candidates.empty_anions",
            _test_candidate_generation_empty_anions,
        ),
        (
            "18.3.candidates.polarity_order",
            _test_candidate_generation_polarity_order,
        ),
        (
            "18.3.detection.valid_pair",
            _test_detect_valid_salt_bridge_pair,
        ),
        (
            "18.3.detection.distant_pair_rejected",
            _test_reject_distant_salt_bridge_pair,
        ),
        (
            "18.3.detection.identifiers",
            _test_detect_pair_preserves_identifiers,
        ),
        (
            "18.3.detection.interaction_id",
            _test_detect_pair_builds_unique_identifier,
        ),
        (
            "18.3.detection.multiple_valid_pairs",
            _test_detect_multiple_valid_pairs,
        ),
        (
            "18.3.detection.exclude_invalid_default",
            _test_detection_excludes_invalid_by_default,
        ),
        (
            "18.3.detection.no_duplicates",
            _test_detection_no_duplicate_interactions,
        ),
        (
            "18.3.classification.strong",
            _test_classify_strong_salt_bridge,
        ),
        (
            "18.3.classification.moderate",
            _test_classify_moderate_salt_bridge,
        ),
        (
            "18.3.classification.weak",
            _test_classify_weak_salt_bridge,
        ),
        (
            "18.3.classification.rejected",
            _test_classify_invalid_as_rejected,
        ),
        (
            "18.3.classification.distance_monotonicity",
            _test_strength_improves_with_shorter_distance,
        ),
        (
            "18.3.scoring.valid_positive",
            _test_valid_interaction_has_positive_score,
        ),
        (
            "18.3.scoring.invalid_zero",
            _test_invalid_interaction_has_zero_or_minimal_score,
        ),
        (
            "18.3.scoring.shorter_higher",
            _test_shorter_interaction_scores_higher,
        ),
        (
            "18.3.scoring.contact_count",
            _test_multiple_contacts_do_not_reduce_score,
        ),
        (
            "18.3.scoring.charge_magnitude",
            _test_stronger_charge_does_not_reduce_score,
        ),
        (
            "18.3.scoring.finite",
            _test_score_is_finite,
        ),
        (
            "18.3.consistency.detected_strength",
            _test_detected_interaction_strength_matches_classifier,
        ),
        (
            "18.3.consistency.detected_score",
            _test_detected_interaction_score_matches_scorer,
        ),
        (
            "18.3.consistency.valid_not_rejected",
            _test_detected_valid_interaction_not_rejected,
        ),
        (
            "18.3.pipeline.positive",
            _test_complete_pipeline_positive_case,
        ),
        (
            "18.3.pipeline.distant_negative",
            _test_complete_pipeline_distant_negative_case,
        ),
        (
            "18.3.pipeline.neutral_negative",
            _test_complete_pipeline_neutral_negative_case,
        ),
        (
            "18.3.pipeline.mixed_pairs",
            _test_complete_pipeline_mixed_pairs,
        ),
        (
            "18.3.pipeline.identifiers",
            _test_complete_pipeline_result_identifiers,
        ),
        (
            "18.3.invariants.polarities",
            _test_all_detected_interactions_have_valid_polarities,
        ),
        (
            "18.3.invariants.finite_scores",
            _test_all_detected_scores_are_finite,
        ),
        (
            "18.3.invariants.geometry",
            _test_all_detected_geometries_are_consistent,
        ),
        (
            "18.3.invariants.valid_positive_scores",
            _test_valid_interactions_have_positive_scores,
        ),
        (
            "18.3.invariants.rejected_not_strong",
            _test_rejected_interactions_are_not_strong,
        ),
    ]


# =============================================================================
# 18.3.12. SECTION RUNNER
# =============================================================================


def run_salt_bridge_detection_classification_tests(
    *,
    report: Optional[
        _SelfTestReport
    ] = None,
    raise_on_failure: bool = False,
    print_report: bool = False,
) -> _SelfTestReport:
    """
    Run all Section 18.3 detection and classification self-tests.

    Parameters
    ----------
    report
        Optional existing self-test report.
    raise_on_failure
        Whether the first failure should raise an exception.
    print_report
        Whether the updated report should be printed.

    Returns
    -------
    _SelfTestReport
        Updated self-test report.
    """

    resolved_report = (
        _run_self_test_group(
            get_salt_bridge_detection_classification_tests(),
            report=report,
            raise_on_failure=(
                raise_on_failure
            ),
        )
    )

    if print_report:
        _print_self_test_report(
            resolved_report
        )

    return resolved_report



# =============================================================================
# 18.4. INTEGRATION AND SERIALIZATION TESTS
# =============================================================================


# =============================================================================
# 18.4.1. INTEGRATION FUNCTION RESOLUTION
# =============================================================================


def _call_attach_salt_bridge_results(
    dock_model: Any,
    result: SaltBridgeResult,
    *,
    preserve_existing: bool = False,
) -> Any:
    """
    Call the available DockModel attachment function.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    result
        Salt-bridge result.
    preserve_existing
        Whether existing interactions should be preserved.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    attachment_function = (
        _resolve_self_test_callable(
            "attach_salt_bridge_results",
            "attach_salt_bridges",
            "set_salt_bridge_results",
            "update_dock_model_salt_bridges",
        )
    )

    call_variants = (
        {
            "preserve_existing": (
                preserve_existing
            ),
        },
        {
            "append": preserve_existing,
        },
        {},
    )

    last_error = None

    for keyword_arguments in (
        call_variants
    ):
        try:
            updated_model = (
                attachment_function(
                    dock_model,
                    result,
                    **keyword_arguments,
                )
            )

            return (
                updated_model
                if updated_model is not None
                else dock_model
            )

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call the DockModel attachment function."
    ) from last_error


def _call_analyze_dock_model_salt_bridges(
    dock_model: Any,
    config: Optional[
        SaltBridgeConfig
    ] = None,
    *,
    preserve_existing: bool = False,
) -> Any:
    """
    Call the available DockModel salt-bridge analysis function.

    Parameters
    ----------
    dock_model
        DockModel-like object.
    config
        Optional configuration.
    preserve_existing
        Whether existing salt bridges should be preserved.

    Returns
    -------
    Any
        Updated DockModel-like object.
    """

    analysis_function = (
        _resolve_self_test_callable(
            "analyze_dock_model_salt_bridges",
            "analyze_dock_model_saltbridge",
            "analyze_salt_bridges_for_dock_model",
        )
    )

    call_variants = (
        {
            "config": config,
            "preserve_existing": (
                preserve_existing
            ),
        },
        {
            "config": config,
        },
        {
            "preserve_existing": (
                preserve_existing
            ),
        },
        {},
    )

    last_error = None

    for keyword_arguments in (
        call_variants
    ):
        filtered_arguments = {
            key: value
            for key, value
            in keyword_arguments.items()
            if value is not None
        }

        try:
            result = analysis_function(
                dock_model,
                **filtered_arguments,
            )

            return (
                result
                if result is not None
                else dock_model
            )

        except TypeError as error:
            last_error = error

    raise SaltBridgeSelfTestError(
        "Could not call DockModel salt-bridge analysis."
    ) from last_error


def _call_analyze_multiple_dock_models_salt_bridges(
    dock_models: Iterable[Any],
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> List[Any]:
    """
    Call the available multiple-DockModel analysis function.

    Parameters
    ----------
    dock_models
        DockModel-like objects.
    config
        Optional configuration.

    Returns
    -------
    List[Any]
        Updated models.
    """

    analysis_function = (
        _resolve_self_test_callable(
            "analyze_multiple_dock_models_salt_bridges",
            "analyze_dock_models_salt_bridges",
            "analyze_multiple_salt_bridge_models",
        )
    )

    try:
        result = analysis_function(
            dock_models,
            config=config,
        )

    except TypeError:
        try:
            result = analysis_function(
                dock_models,
                config,
            )

        except TypeError:
            result = analysis_function(
                dock_models
            )

    return list(
        result
        or []
    )


# =============================================================================
# 18.4.2. STATISTICS FUNCTION RESOLUTION
# =============================================================================


def _call_calculate_result_statistics(
    result: SaltBridgeResult,
    *,
    in_place: bool = False,
) -> Mapping[str, Any]:
    """
    Call the available result-statistics function.

    Parameters
    ----------
    result
        Salt-bridge result.
    in_place
        Whether statistics should be stored in the result.

    Returns
    -------
    Mapping[str, Any]
        Statistics mapping.
    """

    statistics_function = (
        _resolve_self_test_callable(
            "calculate_salt_bridge_result_statistics",
            "calculate_salt_bridge_statistics",
            "summarize_salt_bridge_statistics",
            "build_salt_bridge_statistics",
        )
    )

    try:
        statistics = statistics_function(
            result,
            in_place=in_place,
        )

    except TypeError:
        try:
            statistics = statistics_function(
                result
            )

        except TypeError:
            statistics = statistics_function(
                result.interactions
            )

    if isinstance(
        statistics,
        SaltBridgeResult,
    ):
        statistics = (
            statistics.statistics
        )

    if statistics is None:
        statistics = result.statistics

    if not isinstance(
        statistics,
        Mapping,
    ):
        raise SaltBridgeSelfTestError(
            "Statistics function did not return a mapping."
        )

    return statistics


def _call_build_compact_summary(
    result: SaltBridgeResult,
) -> Mapping[str, Any]:
    """
    Call the available compact-summary function.
    """

    summary_function = (
        _resolve_self_test_callable(
            "build_compact_salt_bridge_summary",
            "build_salt_bridge_compact_summary",
            "summarize_salt_bridge_result",
        )
    )

    summary = summary_function(
        result
    )

    if not isinstance(
        summary,
        Mapping,
    ):
        raise SaltBridgeSelfTestError(
            "Compact summary must be a mapping."
        )

    return summary


def _call_build_text_summary(
    result: SaltBridgeResult,
) -> str:
    """
    Call the available text-summary function.
    """

    summary_function = (
        _resolve_self_test_callable(
            "build_salt_bridge_text_summary",
            "format_salt_bridge_summary",
            "summarize_salt_bridges_text",
        )
    )

    summary = summary_function(
        result
    )

    return str(
        summary
    )


# =============================================================================
# 18.4.3. MULTIPOSE FUNCTION RESOLUTION
# =============================================================================


def _call_analyze_salt_bridges_multipose(
    poses: Iterable[Any],
    config: Optional[
        SaltBridgeConfig
    ] = None,
) -> Mapping[str, Any]:
    """
    Call the complete multipose salt-bridge pipeline.

    Parameters
    ----------
    poses
        Pose-like inputs.
    config
        Optional configuration.

    Returns
    -------
    Mapping[str, Any]
        Multipose analysis.
    """

    multipose_function = (
        _resolve_self_test_callable(
            "analyze_salt_bridges_multipose",
            "analyze_multiple_poses_salt_bridges",
            "run_salt_bridge_multipose_analysis",
        )
    )

    try:
        result = multipose_function(
            poses,
            config=config,
        )

    except TypeError:
        try:
            result = multipose_function(
                poses,
                config,
            )

        except TypeError:
            result = multipose_function(
                poses
            )

    if not isinstance(
        result,
        Mapping,
    ):
        raise SaltBridgeSelfTestError(
            "Multipose analysis must return a mapping."
        )

    return result


def _call_rank_salt_bridge_poses(
    results: Iterable[
        SaltBridgeResult
    ],
) -> List[Mapping[str, Any]]:
    """
    Call the available pose-ranking function.
    """

    ranking_function = (
        _resolve_self_test_callable(
            "rank_salt_bridge_poses",
            "rank_poses_by_salt_bridges",
            "rank_salt_bridge_results",
        )
    )

    ranking = ranking_function(
        results
    )

    return list(
        ranking
        or []
    )


def _call_calculate_persistence(
    results: Iterable[
        SaltBridgeResult
    ],
) -> List[Mapping[str, Any]]:
    """
    Call the available interaction-persistence function.
    """

    persistence_function = (
        _resolve_self_test_callable(
            "calculate_salt_bridge_persistence",
            "calculate_interaction_persistence",
            "build_salt_bridge_persistence",
            "summarize_salt_bridge_persistence",
        )
    )

    persistence = persistence_function(
        results
    )

    if isinstance(
        persistence,
        Mapping,
    ):
        persistence = (
            persistence.get(
                "persistence"
            )
            or persistence.get(
                "records"
            )
            or persistence.get(
                "interactions"
            )
            or []
        )

    return list(
        persistence
        or []
    )


def _call_build_consensus_interactions(
    results: Iterable[
        SaltBridgeResult
    ],
) -> List[Mapping[str, Any]]:
    """
    Call the available multipose consensus function.
    """

    consensus_function = (
        _resolve_self_test_callable(
            "build_consensus_salt_bridge_interactions",
            "build_salt_bridge_consensus",
            "calculate_salt_bridge_consensus",
            "identify_consensus_salt_bridges",
        )
    )

    consensus = consensus_function(
        results
    )

    if isinstance(
        consensus,
        Mapping,
    ):
        consensus = (
            consensus.get(
                "consensus_interactions"
            )
            or consensus.get(
                "consensus"
            )
            or consensus.get(
                "records"
            )
            or []
        )

    return list(
        consensus
        or []
    )


# =============================================================================
# 18.4.4. DOCKMODEL ATTACHMENT TESTS
# =============================================================================


def _test_attach_result_to_dock_model() -> None:
    """
    Test attachment of detected interactions to DockModel.
    """

    result = _make_test_result()

    dock_model = _MockDockModel(
        source=[]
    )

    updated_model = (
        _call_attach_salt_bridge_results(
            dock_model,
            result,
        )
    )

    _assert_is_instance(
        updated_model.saltbridge,
        list,
    )

    _assert_length(
        updated_model.saltbridge,
        len(
            result.interactions
        ),
    )

    _assert_equal(
        updated_model.saltbridge,
        result.interactions,
    )


def _test_attach_result_replaces_existing_by_default() -> None:
    """
    Test default replacement of pre-existing salt bridges.
    """

    previous_interaction = (
        _make_test_interaction(
            interaction_id="previous"
        )
    )

    new_interaction = (
        _make_test_interaction(
            interaction_id="new"
        )
    )

    dock_model = _MockDockModel(
        source=[],
        saltbridge=[
            previous_interaction
        ],
    )

    result = _make_test_result(
        interactions=[
            new_interaction
        ]
    )

    updated_model = (
        _call_attach_salt_bridge_results(
            dock_model,
            result,
            preserve_existing=False,
        )
    )

    _assert_length(
        updated_model.saltbridge,
        1,
    )

    _assert_equal(
        updated_model.saltbridge[0]
        .interaction_id,
        "new",
    )


def _test_attach_result_preserves_existing_when_requested() -> None:
    """
    Test preservation of pre-existing interactions when supported.
    """

    previous_interaction = (
        _make_test_interaction(
            interaction_id="previous"
        )
    )

    new_interaction = (
        _make_test_interaction(
            interaction_id="new"
        )
    )

    dock_model = _MockDockModel(
        source=[],
        saltbridge=[
            previous_interaction
        ],
    )

    result = _make_test_result(
        interactions=[
            new_interaction
        ]
    )

    updated_model = (
        _call_attach_salt_bridge_results(
            dock_model,
            result,
            preserve_existing=True,
        )
    )

    interaction_ids = {
        interaction.interaction_id
        for interaction
        in updated_model.saltbridge
    }

    _assert_contains(
        interaction_ids,
        "previous",
    )

    _assert_contains(
        interaction_ids,
        "new",
    )


def _test_attachment_does_not_add_dynamic_result_fields() -> None:
    """
    Test the simplified DockModel architecture.

    Only the ``saltbridge`` field should be required.
    """

    dock_model = _MockDockModel(
        source=[]
    )

    updated_model = (
        _call_attach_salt_bridge_results(
            dock_model,
            _make_test_result(),
        )
    )

    _assert_true(
        hasattr(
            updated_model,
            "saltbridge",
        )
    )

    for deprecated_field in (
        "saltbridge_score",
        "saltbridge_statistics",
        "saltbridge_summary",
        "saltbridge_result",
    ):
        _assert_false(
            hasattr(
                updated_model,
                deprecated_field,
            ),
            (
                f"DockModel should not require "
                f"{deprecated_field!r}."
            ),
        )


# =============================================================================
# 18.4.5. DOCKMODEL ANALYSIS TESTS
# =============================================================================


def _test_analyze_dock_model_salt_bridges() -> None:
    """
    Test complete analysis and attachment for one DockModel.
    """

    source = [
        _make_mock_lysine(),
        _make_mock_aspartate(),
    ]

    dock_model = _MockDockModel(
        source=source,
        pose_id=1,
        model_id="model_1",
    )

    updated_model = (
        _call_analyze_dock_model_salt_bridges(
            dock_model
        )
    )

    _assert_is_instance(
        updated_model.saltbridge,
        list,
    )

    _assert_not_empty(
        updated_model.saltbridge,
        (
            "DockModel analysis should attach "
            "at least one interaction."
        ),
    )


def _test_analyze_multiple_dock_models() -> None:
    """
    Test analysis of multiple DockModel instances.
    """

    models = [
        _MockDockModel(
            source=[
                _make_mock_lysine(
                    number=10
                ),
                _make_mock_aspartate(
                    number=40
                ),
            ],
            pose_id=1,
            model_id="model_1",
        ),
        _MockDockModel(
            source=[
                _make_mock_arginine(
                    number=20
                ),
                _make_mock_glutamate(
                    number=50
                ),
            ],
            pose_id=2,
            model_id="model_2",
        ),
    ]

    analyzed_models = (
        _call_analyze_multiple_dock_models_salt_bridges(
            models
        )
    )

    _assert_length(
        analyzed_models,
        2,
    )

    for model in analyzed_models:
        _assert_is_instance(
            model.saltbridge,
            list,
        )


def _test_dock_model_empty_source() -> None:
    """
    Test DockModel analysis with no molecular source content.
    """

    dock_model = _MockDockModel(
        source=[]
    )

    updated_model = (
        _call_analyze_dock_model_salt_bridges(
            dock_model
        )
    )

    _assert_is_instance(
        updated_model.saltbridge,
        list,
    )

    _assert_empty(
        updated_model.saltbridge
    )


# =============================================================================
# 18.4.6. STATISTICS TESTS
# =============================================================================


def _make_statistics_test_result(
) -> SaltBridgeResult:
    """
    Create a result containing strong, moderate, and weak interactions.
    """

    interactions = [
        _make_test_interaction(
            interaction_id="statistics_strong",
            strength=STRENGTH_STRONG,
            score=1.5,
            geometry=_make_test_geometry(
                center_distance=2.8,
                minimum_atom_distance=2.5,
                maximum_atom_distance=3.1,
                mean_atom_distance=2.8,
                contact_count=3,
                valid=True,
            ),
        ),
        _make_test_interaction(
            interaction_id="statistics_moderate",
            strength=STRENGTH_MODERATE,
            score=1.0,
            geometry=_make_test_geometry(
                center_distance=3.5,
                minimum_atom_distance=3.1,
                maximum_atom_distance=3.9,
                mean_atom_distance=3.5,
                contact_count=2,
                valid=True,
            ),
        ),
        _make_test_interaction(
            interaction_id="statistics_weak",
            strength=STRENGTH_WEAK,
            score=0.5,
            geometry=_make_test_geometry(
                center_distance=4.2,
                minimum_atom_distance=3.9,
                maximum_atom_distance=4.5,
                mean_atom_distance=4.2,
                contact_count=1,
                valid=True,
            ),
        ),
    ]

    return _make_test_result(
        interactions=interactions
    )


def _test_statistics_interaction_count() -> None:
    """
    Test total interaction count.
    """

    result = (
        _make_statistics_test_result()
    )

    statistics = (
        _call_calculate_result_statistics(
            result
        )
    )

    interaction_count = (
        statistics.get(
            "interaction_count"
        )
        or statistics.get(
            "total_interactions"
        )
        or statistics.get(
            "count"
        )
    )

    _assert_equal(
        interaction_count,
        3,
    )


def _test_statistics_total_score() -> None:
    """
    Test total interaction score.
    """

    result = (
        _make_statistics_test_result()
    )

    statistics = (
        _call_calculate_result_statistics(
            result
        )
    )

    total_score = (
        statistics.get(
            "total_score"
        )
        or statistics.get(
            "score_total"
        )
    )

    _assert_almost_equal(
        total_score,
        3.0,
    )


def _test_statistics_distance_extrema() -> None:
    """
    Test minimum and maximum distance statistics.
    """

    statistics = (
        _call_calculate_result_statistics(
            _make_statistics_test_result()
        )
    )

    minimum_distance = (
        statistics.get(
            "minimum_distance"
        )
        or statistics.get(
            "min_distance"
        )
    )

    maximum_distance = (
        statistics.get(
            "maximum_distance"
        )
        or statistics.get(
            "max_distance"
        )
    )

    _assert_almost_equal(
        minimum_distance,
        2.5,
    )

    _assert_almost_equal(
        maximum_distance,
        4.5,
    )


def _test_statistics_strength_distribution() -> None:
    """
    Test distribution by interaction strength.
    """

    statistics = (
        _call_calculate_result_statistics(
            _make_statistics_test_result()
        )
    )

    strength_counts = (
        statistics.get(
            "strength_counts"
        )
        or statistics.get(
            "strength_distribution"
        )
        or {}
    )

    _assert_equal(
        strength_counts.get(
            STRENGTH_STRONG,
            0,
        ),
        1,
    )

    _assert_equal(
        strength_counts.get(
            STRENGTH_MODERATE,
            0,
        ),
        1,
    )

    _assert_equal(
        strength_counts.get(
            STRENGTH_WEAK,
            0,
        ),
        1,
    )


def _test_statistics_in_place_update() -> None:
    """
    Test storage of calculated statistics in SaltBridgeResult.
    """

    result = (
        _make_statistics_test_result()
    )

    _call_calculate_result_statistics(
        result,
        in_place=True,
    )

    _assert_not_empty(
        result.statistics,
        (
            "In-place statistics calculation "
            "should update result.statistics."
        ),
    )


def _test_empty_result_statistics() -> None:
    """
    Test statistics for an empty result.
    """

    result = _make_test_result(
        interactions=[],
        cationic_groups=[],
        anionic_groups=[],
    )

    statistics = (
        _call_calculate_result_statistics(
            result
        )
    )

    interaction_count = (
        statistics.get(
            "interaction_count",
            statistics.get(
                "total_interactions",
                0,
            ),
        )
    )

    _assert_equal(
        interaction_count,
        0,
    )


# =============================================================================
# 18.4.7. SUMMARY TESTS
# =============================================================================


def _test_compact_summary_structure() -> None:
    """
    Test required compact-summary fields.
    """

    summary = (
        _call_build_compact_summary(
            _make_statistics_test_result()
        )
    )

    _assert_mapping_contains_keys(
        summary,
        (
            "interaction_count",
            "total_score",
        ),
    )


def _test_compact_summary_values() -> None:
    """
    Test compact-summary values.
    """

    summary = (
        _call_build_compact_summary(
            _make_statistics_test_result()
        )
    )

    _assert_equal(
        summary[
            "interaction_count"
        ],
        3,
    )

    _assert_almost_equal(
        summary[
            "total_score"
        ],
        3.0,
    )


def _test_text_summary_nonempty() -> None:
    """
    Test generation of a non-empty text summary.
    """

    summary = (
        _call_build_text_summary(
            _make_statistics_test_result()
        )
    )

    _assert_true(
        bool(
            summary.strip()
        ),
        "Text summary should not be empty.",
    )


def _test_empty_result_text_summary() -> None:
    """
    Test text summary generation for an empty result.
    """

    result = _make_test_result(
        interactions=[],
        cationic_groups=[],
        anionic_groups=[],
    )

    summary = (
        _call_build_text_summary(
            result
        )
    )

    _assert_is_instance(
        summary,
        str,
    )


# =============================================================================
# 18.4.8. MULTIPOSE RANKING TESTS
# =============================================================================


def _test_pose_ranking_count() -> None:
    """
    Test one ranking record per pose.
    """

    results = _make_test_pose_results(
        pose_count=3
    )

    ranking = (
        _call_rank_salt_bridge_poses(
            results
        )
    )

    _assert_length(
        ranking,
        3,
    )


def _test_pose_ranking_order() -> None:
    """
    Test descending ranking-score order.
    """

    ranking = (
        _call_rank_salt_bridge_poses(
            _make_test_pose_results(
                pose_count=3
            )
        )
    )

    ranking_scores = [
        safe_float(
            record.get(
                "ranking_score",
                record.get(
                    "score"
                ),
            ),
            default=0.0,
        )
        or 0.0
        for record in ranking
    ]

    _assert_equal(
        ranking_scores,
        sorted(
            ranking_scores,
            reverse=True,
        ),
        (
            "Pose ranking should be ordered "
            "by decreasing score."
        ),
    )


def _test_pose_ranking_unique_ranks() -> None:
    """
    Test unique sequential pose ranks.
    """

    ranking = (
        _call_rank_salt_bridge_poses(
            _make_test_pose_results(
                pose_count=3
            )
        )
    )

    ranks = [
        safe_int(
            record.get(
                "rank"
            )
        )
        for record in ranking
    ]

    _assert_equal(
        sorted(
            ranks
        ),
        [
            1,
            2,
            3,
        ],
    )


# =============================================================================
# 18.4.9. PERSISTENCE AND CONSENSUS TESTS
# =============================================================================


def _make_persistent_pose_results(
) -> List[SaltBridgeResult]:
    """
    Create pose results sharing the same residue-level interaction.
    """

    results: List[
        SaltBridgeResult
    ] = []

    for pose_id in (
        1,
        2,
        3,
    ):
        cation_residue = (
            _make_mock_lysine(
                number=10,
                chain_id="A",
            )
        )

        anion_residue = (
            _make_mock_aspartate(
                number=40,
                chain_id="B",
            )
        )

        cation = (
            _make_test_cation_group(
                group_id=(
                    f"persistent_cation_{pose_id}"
                ),
                residue=(
                    cation_residue
                ),
            )
        )

        anion = (
            _make_test_anion_group(
                group_id=(
                    f"persistent_anion_{pose_id}"
                ),
                residue=(
                    anion_residue
                ),
            )
        )

        interaction = (
            _make_test_interaction(
                interaction_id=(
                    f"persistent_{pose_id}"
                ),
                cation=cation,
                anion=anion,
                score=1.0,
                pose_id=pose_id,
                model_id=(
                    f"model_{pose_id}"
                ),
            )
        )

        results.append(
            _make_test_result(
                interactions=[
                    interaction
                ],
                pose_id=pose_id,
                model_id=(
                    f"model_{pose_id}"
                ),
            )
        )

    return results


def _test_persistence_records_nonempty() -> None:
    """
    Test persistence calculation across repeated poses.
    """

    persistence = (
        _call_calculate_persistence(
            _make_persistent_pose_results()
        )
    )

    _assert_not_empty(
        persistence,
        (
            "Repeated residue-level interactions "
            "should generate persistence records."
        ),
    )


def _test_persistence_fraction_range() -> None:
    """
    Test persistence fractions remain between zero and one.
    """

    persistence = (
        _call_calculate_persistence(
            _make_persistent_pose_results()
        )
    )

    for record in persistence:
        persistence_value = (
            record.get(
                "persistence"
            )
            or record.get(
                "persistence_fraction"
            )
            or record.get(
                "frequency"
            )
        )

        if persistence_value is None:
            continue

        _assert_between(
            persistence_value,
            0.0,
            1.0,
        )


def _test_fully_persistent_interaction() -> None:
    """
    Test identification of an interaction present in every pose.
    """

    persistence = (
        _call_calculate_persistence(
            _make_persistent_pose_results()
        )
    )

    persistence_values = [
        safe_float(
            record.get(
                "persistence",
                record.get(
                    "persistence_fraction",
                    record.get(
                        "frequency"
                    ),
                ),
            )
        )
        for record in persistence
    ]

    persistence_values = [
        value
        for value
        in persistence_values
        if value is not None
    ]

    _assert_true(
        any(
            math.isclose(
                value,
                1.0,
                abs_tol=1e-6,
            )
            for value
            in persistence_values
        ),
        (
            "At least one interaction should have "
            "full persistence."
        ),
    )


def _test_consensus_records_nonempty() -> None:
    """
    Test consensus generation across repeated poses.
    """

    consensus = (
        _call_build_consensus_interactions(
            _make_persistent_pose_results()
        )
    )

    _assert_not_empty(
        consensus
    )


def _test_consensus_is_json_safe() -> None:
    """
    Test JSON compatibility of consensus records.
    """

    consensus = (
        _call_build_consensus_interactions(
            _make_persistent_pose_results()
        )
    )

    _assert_json_serializable(
        make_json_safe(
            consensus
        )
    )


# =============================================================================
# 18.4.10. COMPLETE MULTIPOSE PIPELINE TESTS
# =============================================================================


def _test_complete_multipose_analysis() -> None:
    """
    Test the complete multipose analysis pipeline.
    """

    poses = [
        [
            _make_mock_lysine(
                number=10
            ),
            _make_mock_aspartate(
                number=40,
                center=(
                    3.0,
                    0.0,
                    0.0,
                ),
            ),
        ],
        [
            _make_mock_lysine(
                number=10
            ),
            _make_mock_aspartate(
                number=40,
                center=(
                    3.2,
                    0.0,
                    0.0,
                ),
            ),
        ],
    ]

    multipose_result = (
        _call_analyze_salt_bridges_multipose(
            poses
        )
    )

    _assert_mapping_contains_keys(
        multipose_result,
        (
            "results",
        ),
    )

    _assert_length(
        multipose_result[
            "results"
        ],
        2,
    )


def _test_empty_multipose_analysis() -> None:
    """
    Test multipose analysis with no poses.
    """

    multipose_result = (
        _call_analyze_salt_bridges_multipose(
            []
        )
    )

    results = (
        multipose_result.get(
            "results",
            [],
        )
    )

    _assert_empty(
        results
    )


def _test_multipose_results_have_pose_ids() -> None:
    """
    Test pose identifier assignment in multipose analysis.
    """

    multipose_result = (
        _call_analyze_salt_bridges_multipose(
            [
                [
                    _make_mock_lysine(),
                    _make_mock_aspartate(),
                ],
                [
                    _make_mock_arginine(),
                    _make_mock_glutamate(),
                ],
            ]
        )
    )

    results = list(
        multipose_result.get(
            "results",
            [],
        )
    )

    _assert_length(
        results,
        2,
    )

    resolved_pose_ids = [
        result.pose_id
        for result in results
    ]

    _assert_true(
        all(
            pose_id is not None
            for pose_id
            in resolved_pose_ids
        ),
        (
            "Multipose results should receive "
            "pose identifiers."
        ),
    )


# =============================================================================
# 18.4.11. BASIC SERIALIZATION TESTS
# =============================================================================


def _test_charged_atom_serialization() -> None:
    """
    Test ChargedAtom dictionary serialization.
    """

    charged_atom = (
        _make_test_charged_atom(
            name="N1",
            element="N",
            coordinate=(
                1.0,
                2.0,
                3.0,
            ),
            polarity="positive",
            effective_charge=1.0,
        )
    )

    serialized = (
        charged_atom_to_dict(
            charged_atom
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "name",
            "element",
            "coordinate",
            "effective_charge",
            "polarity",
        ),
    )

    _assert_json_serializable(
        serialized
    )


def _test_charged_group_serialization() -> None:
    """
    Test ChargedGroup dictionary serialization.
    """

    serialized = (
        charged_group_to_dict(
            _make_test_anion_group()
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "group_id",
            "group_type",
            "polarity",
            "center",
            "net_charge",
            "atoms",
        ),
    )

    _assert_json_serializable(
        serialized
    )


def _test_geometry_serialization() -> None:
    """
    Test SaltBridgeGeometry dictionary serialization.
    """

    serialized = (
        salt_bridge_geometry_to_dict(
            _make_test_geometry()
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "center_distance",
            "minimum_atom_distance",
            "maximum_atom_distance",
            "mean_atom_distance",
            "contact_count",
            "valid",
        ),
    )

    _assert_json_serializable(
        serialized
    )


def _test_interaction_serialization() -> None:
    """
    Test SaltBridgeInteraction dictionary serialization.
    """

    serialized = (
        salt_bridge_interaction_to_dict(
            _make_test_interaction()
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "interaction_id",
            "interaction_type",
            "strength",
            "score",
            "cation",
            "anion",
            "geometry",
        ),
    )

    _assert_json_serializable(
        serialized
    )


def _test_compact_interaction_serialization() -> None:
    """
    Test compact interaction serialization.
    """

    serialized = (
        salt_bridge_interaction_to_dict(
            _make_test_interaction(),
            compact=True,
        )
    )

    _assert_is_instance(
        serialized,
        Mapping,
    )

    _assert_json_serializable(
        serialized
    )


def _test_result_serialization() -> None:
    """
    Test SaltBridgeResult dictionary serialization.
    """

    serialized = (
        salt_bridge_result_to_dict(
            _make_test_result()
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "schema",
            "schema_version",
            "interaction_count",
            "interactions",
            "statistics",
            "metadata",
        ),
    )

    _assert_equal(
        serialized[
            "schema"
        ],
        "dockanalyzer.saltbridge",
    )

    _assert_json_serializable(
        serialized
    )


# =============================================================================
# 18.4.12. STRICT JSON TESTS
# =============================================================================


def _test_result_json_serialization() -> None:
    """
    Test serialization of SaltBridgeResult to JSON text.
    """

    json_document = (
        serialize_salt_bridge_result(
            _make_test_result()
        )
    )

    _assert_is_instance(
        json_document,
        str,
    )

    parsed_document = json.loads(
        json_document
    )

    _assert_is_instance(
        parsed_document,
        Mapping,
    )


def _test_result_json_rejects_nonfinite_values() -> None:
    """
    Test conversion of non-finite values to JSON null.
    """

    result = _make_test_result()

    result.metadata[
        "nan_value"
    ] = float(
        "nan"
    )

    result.metadata[
        "positive_infinity"
    ] = float(
        "inf"
    )

    result.metadata[
        "negative_infinity"
    ] = float(
        "-inf"
    )

    json_document = (
        serialize_salt_bridge_result(
            result
        )
    )

    parsed_document = json.loads(
        json_document
    )

    metadata = parsed_document[
        "metadata"
    ]

    _assert_is_none(
        metadata[
            "nan_value"
        ]
    )

    _assert_is_none(
        metadata[
            "positive_infinity"
        ]
    )

    _assert_is_none(
        metadata[
            "negative_infinity"
        ]
    )


def _test_make_json_safe_nested_data() -> None:
    """
    Test recursive conversion of nested values.
    """

    nested_value = {
        (
            "tuple",
            1,
        ): {
            "set": {
                3,
                2,
                1,
            },
            "values": [
                float(
                    "nan"
                ),
                5.0,
            ],
        }
    }

    serialized = make_json_safe(
        nested_value
    )

    _assert_json_serializable(
        serialized
    )


def _test_interactions_json_serialization() -> None:
    """
    Test JSON serialization of interaction collections.
    """

    json_document = (
        serialize_salt_bridge_interactions(
            [
                _make_test_interaction()
            ]
        )
    )

    parsed_document = json.loads(
        json_document
    )

    _assert_is_instance(
        parsed_document,
        list,
    )

    _assert_length(
        parsed_document,
        1,
    )


# =============================================================================
# 18.4.13. TABLE-EXPORT TESTS
# =============================================================================


def _test_interaction_rows() -> None:
    """
    Test generation of flat interaction rows.
    """

    rows = (
        salt_bridge_interactions_to_rows(
            [
                _make_test_interaction()
            ]
        )
    )

    _assert_length(
        rows,
        1,
    )

    _assert_is_instance(
        rows[0],
        Mapping,
    )

    _assert_json_serializable(
        rows
    )


def _test_group_rows() -> None:
    """
    Test generation of flat charged-group rows.
    """

    rows = salt_bridge_groups_to_rows(
        [
            _make_test_cation_group(),
            _make_test_anion_group(),
        ]
    )

    _assert_length(
        rows,
        2,
    )

    _assert_mapping_contains_keys(
        rows[0],
        (
            "group_id",
            "group_type",
            "polarity",
            "net_charge",
            "center_x",
            "center_y",
            "center_z",
        ),
    )


def _test_statistics_rows() -> None:
    """
    Test flattening of nested statistics.
    """

    rows = (
        salt_bridge_statistics_to_rows(
            {
                "interaction_count": 2,
                "distances": {
                    "minimum": 2.5,
                    "maximum": 4.0,
                },
            }
        )
    )

    metrics = {
        row[
            "metric"
        ]
        for row in rows
    }

    _assert_contains(
        metrics,
        "interaction_count",
    )

    _assert_contains(
        metrics,
        "distances.minimum",
    )

    _assert_contains(
        metrics,
        "distances.maximum",
    )


def _test_residue_summary_rows() -> None:
    """
    Test residue-level summary export.
    """

    rows = (
        build_residue_salt_bridge_summary_rows(
            _make_test_result()
            .interactions
        )
    )

    _assert_not_empty(
        rows
    )

    _assert_json_serializable(
        rows
    )


def _test_pose_summary_rows() -> None:
    """
    Test pose-level summary export.
    """

    rows = (
        build_pose_salt_bridge_summary_rows(
            _make_test_pose_results(
                pose_count=3
            )
        )
    )

    _assert_length(
        rows,
        3,
    )

    _assert_equal(
        [
            row[
                "rank"
            ]
            for row in rows
        ],
        [
            1,
            2,
            3,
        ],
    )


# =============================================================================
# 18.4.14. EXPORT-PAYLOAD TESTS
# =============================================================================


def _test_single_result_export_payload() -> None:
    """
    Test complete export payload for one result.
    """

    payload = (
        build_salt_bridge_export_payload(
            _make_statistics_test_result()
        )
    )

    _assert_mapping_contains_keys(
        payload,
        (
            "schema",
            "summary",
            "text_summary",
            "result",
            "interaction_rows",
            "cationic_group_rows",
            "anionic_group_rows",
            "residue_summary_rows",
            "statistics_rows",
        ),
    )

    _assert_json_serializable(
        payload
    )


def _test_prepare_export_dict() -> None:
    """
    Test generic dictionary export routing.
    """

    exported = prepare_salt_bridge_export(
        _make_test_result(),
        export_format="dict",
    )

    _assert_is_instance(
        exported,
        Mapping,
    )


def _test_prepare_export_json() -> None:
    """
    Test generic JSON export routing.
    """

    exported = prepare_salt_bridge_export(
        _make_test_result(),
        export_format="json",
    )

    _assert_is_instance(
        exported,
        str,
    )

    json.loads(
        exported
    )


def _test_prepare_export_rows() -> None:
    """
    Test generic table-row export routing.
    """

    exported = prepare_salt_bridge_export(
        _make_test_result(),
        export_format="rows",
    )

    _assert_is_instance(
        exported,
        list,
    )


def _test_prepare_export_invalid_format() -> None:
    """
    Test rejection of unsupported export formats.
    """

    _assert_raises(
        SaltBridgeSerializationError,
        prepare_salt_bridge_export,
        _make_test_result(),
        "unsupported_format",
    )


# =============================================================================
# 18.4.15. MULTIPOSE SERIALIZATION TESTS
# =============================================================================


def _make_test_multipose_mapping(
) -> Dict[str, Any]:
    """
    Create a complete multipose mapping for serialization tests.
    """

    results = _make_persistent_pose_results()

    ranking = (
        _call_rank_salt_bridge_poses(
            results
        )
    )

    persistence = (
        _call_calculate_persistence(
            results
        )
    )

    consensus = (
        _call_build_consensus_interactions(
            results
        )
    )

    return {
        "results": results,
        "pose_ranking": ranking,
        "persistence": persistence,
        "consensus_interactions": (
            consensus
        ),
        "statistics": {
            "pose_count": len(
                results
            ),
        },
        "compact_summary": {
            "pose_count": len(
                results
            ),
            "interaction_count": sum(
                len(
                    result.interactions
                )
                for result in results
            ),
        },
        "text_summary": (
            "Test multipose salt-bridge analysis."
        ),
        "warnings": [],
        "metadata": {
            "self_test": True,
        },
    }


def _test_multipose_dictionary_serialization() -> None:
    """
    Test dictionary serialization of multipose results.
    """

    serialized = (
        salt_bridge_multipose_to_dict(
            _make_test_multipose_mapping()
        )
    )

    _assert_mapping_contains_keys(
        serialized,
        (
            "schema",
            "results",
            "pose_ranking",
            "persistence",
            "consensus_interactions",
            "statistics",
        ),
    )

    _assert_json_serializable(
        serialized
    )


def _test_multipose_json_serialization() -> None:
    """
    Test JSON serialization of multipose results.
    """

    json_document = (
        serialize_salt_bridge_multipose(
            _make_test_multipose_mapping()
        )
    )

    parsed_document = json.loads(
        json_document
    )

    _assert_equal(
        parsed_document[
            "schema"
        ],
        (
            "dockanalyzer.saltbridge."
            "multipose"
        ),
    )


def _test_multipose_export_payload() -> None:
    """
    Test complete multipose export payload.
    """

    payload = (
        build_multipose_salt_bridge_export_payload(
            _make_test_multipose_mapping()
        )
    )

    _assert_mapping_contains_keys(
        payload,
        (
            "summary",
            "pose_summary_rows",
            "interaction_rows",
            "persistence_rows",
            "consensus_rows",
            "statistics",
        ),
    )

    _assert_json_serializable(
        payload
    )


# =============================================================================
# 18.4.16. CHIMERAX SPECIFICATION TESTS
# =============================================================================


def _make_chimerax_test_interaction(
) -> SaltBridgeInteraction:
    """
    Create an interaction containing ChimeraX-like model references.
    """

    structure = _make_test_structure(
        model_id=1
    )

    lysine = _make_mock_lysine(
        number=10,
        chain_id="A",
        structure=structure,
    )

    aspartate = (
        _make_mock_aspartate(
            number=40,
            chain_id="B",
            structure=structure,
        )
    )

    cation = _make_test_cation_group(
        group_id="chimerax_cation",
        residue=lysine,
    )

    anion = _make_test_anion_group(
        group_id="chimerax_anion",
        residue=aspartate,
    )

    return _make_test_interaction(
        cation=cation,
        anion=anion,
    )


def _test_atom_chimerax_spec() -> None:
    """
    Test generation of a ChimeraX atom specification.
    """

    residue = _make_mock_lysine(
        number=10,
        chain_id="A",
        structure=_make_test_structure(
            model_id=1
        ),
    )

    atom = residue.atoms[0]

    atom_spec = atom_to_chimerax_spec(
        atom
    )

    _assert_is_not_none(
        atom_spec
    )

    _assert_contains(
        atom_spec,
        "#1",
    )

    _assert_contains(
        atom_spec,
        "/A",
    )

    _assert_contains(
        atom_spec,
        ":10",
    )

    _assert_contains(
        atom_spec,
        "@NZ",
    )


def _test_residue_chimerax_spec() -> None:
    """
    Test generation of a ChimeraX residue specification.
    """

    residue = _make_mock_aspartate(
        number=40,
        chain_id="B",
        structure=_make_test_structure(
            model_id=2
        ),
    )

    residue_spec = (
        residue_to_chimerax_spec(
            residue
        )
    )

    _assert_equal(
        residue_spec,
        "#2/B:40",
    )


def _test_interaction_chimerax_spec() -> None:
    """
    Test generation of an interaction selection specification.
    """

    interaction_spec = (
        salt_bridge_interaction_to_chimerax_spec(
            _make_chimerax_test_interaction()
        )
    )

    _assert_is_not_none(
        interaction_spec
    )

    _assert_contains(
        interaction_spec,
        "#1",
    )


def _test_result_selection_command() -> None:
    """
    Test ChimeraX selection command generation.
    """

    result = _make_test_result(
        interactions=[
            _make_chimerax_test_interaction()
        ]
    )

    command = (
        build_select_salt_bridge_command(
            result
        )
    )

    _assert_is_not_none(
        command
    )

    _assert_true(
        command.startswith(
            "select clear; select "
        )
    )


def _test_pseudobond_command_generation() -> None:
    """
    Test ChimeraX pseudobond command generation.
    """

    command = (
        build_create_salt_bridge_pseudobond_command(
            _make_chimerax_test_interaction()
        )
    )

    _assert_contains(
        command,
        "pbond ",
    )

    _assert_contains(
        command,
        "color ",
    )

    _assert_contains(
        command,
        "radius ",
    )

    _assert_contains(
        command,
        "name ",
    )


def _test_visualization_command_generation() -> None:
    """
    Test complete ChimeraX visualization command generation.
    """

    result = _make_test_result(
        interactions=[
            _make_chimerax_test_interaction()
        ]
    )

    commands = (
        build_salt_bridge_visualization_commands(
            result
        )
    )

    _assert_not_empty(
        commands
    )

    combined_commands = " ".join(
        commands
    )

    _assert_contains(
        combined_commands,
        "pbond",
    )

    _assert_contains(
        combined_commands,
        "select",
    )


def _test_chimerax_record_serialization() -> None:
    """
    Test ChimeraX-oriented export records.
    """

    record = (
        build_chimerax_salt_bridge_record(
            _make_chimerax_test_interaction()
        )
    )

    _assert_mapping_contains_keys(
        record,
        (
            "interaction_id",
            "positive_atom_spec",
            "negative_atom_spec",
            "selection_spec",
            "pseudobond_group",
            "color",
        ),
    )

    _assert_json_serializable(
        record
    )


def _test_chimerax_generation_without_runtime() -> None:
    """
    Test that command construction does not require ChimeraX imports.
    """

    result = _make_test_result(
        interactions=[
            _make_chimerax_test_interaction()
        ]
    )

    commands = (
        build_salt_bridge_visualization_commands(
            result
        )
    )

    _assert_is_instance(
        commands,
        list,
    )


# =============================================================================
# 18.4.17. SERIALIZATION ROUND-TRIP TESTS
# =============================================================================


def _test_result_json_round_trip_core_fields() -> None:
    """
    Test preservation of core fields through JSON serialization.
    """

    result = _make_test_result(
        pose_id=5,
        model_id="model_5",
    )

    parsed = json.loads(
        serialize_salt_bridge_result(
            result
        )
    )

    _assert_equal(
        parsed[
            "pose_id"
        ],
        5,
    )

    _assert_equal(
        parsed[
            "model_id"
        ],
        "model_5",
    )

    _assert_equal(
        parsed[
            "interaction_count"
        ],
        len(
            result.interactions
        ),
    )


def _test_compact_and_full_serialization_counts_match() -> None:
    """
    Test consistency between compact and full result serialization.
    """

    result = _make_test_result()

    full = salt_bridge_result_to_dict(
        result,
        compact_interactions=False,
    )

    compact = salt_bridge_result_to_dict(
        result,
        compact_interactions=True,
        include_groups=False,
    )

    _assert_equal(
        full[
            "interaction_count"
        ],
        compact[
            "interaction_count"
        ],
    )

    _assert_equal(
        len(
            full[
                "interactions"
            ]
        ),
        len(
            compact[
                "interactions"
            ]
        ),
    )


def _test_serialization_does_not_include_raw_atom_objects() -> None:
    """
    Test that serialized output does not retain raw mock atoms.
    """

    serialized = (
        salt_bridge_result_to_dict(
            _make_test_result()
        )
    )

    def contains_mock_atom(
        value: Any,
    ) -> bool:
        if isinstance(
            value,
            _MockAtom,
        ):
            return True

        if isinstance(
            value,
            Mapping,
        ):
            return any(
                contains_mock_atom(
                    nested_value
                )
                for nested_value
                in value.values()
            )

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return any(
                contains_mock_atom(
                    nested_value
                )
                for nested_value
                in value
            )

        return False

    _assert_false(
        contains_mock_atom(
            serialized
        ),
        (
            "Serialized result must not contain "
            "raw molecular objects."
        ),
    )


# =============================================================================
# 18.4.18. SECTION TEST REGISTRY
# =============================================================================


def get_salt_bridge_integration_serialization_tests(
) -> List[
    Tuple[
        str,
        Callable[
            [],
            Any,
        ],
    ]
]:
    """
    Return all Section 18.4 self-tests.

    Returns
    -------
    List[Tuple[str, Callable[[], Any]]]
        Named integration and serialization tests.
    """

    return [
        (
            "18.4.dockmodel.attach",
            _test_attach_result_to_dock_model,
        ),
        (
            "18.4.dockmodel.replace_existing",
            _test_attach_result_replaces_existing_by_default,
        ),
        (
            "18.4.dockmodel.preserve_existing",
            _test_attach_result_preserves_existing_when_requested,
        ),
        (
            "18.4.dockmodel.simplified_architecture",
            _test_attachment_does_not_add_dynamic_result_fields,
        ),
        (
            "18.4.dockmodel.analyze_single",
            _test_analyze_dock_model_salt_bridges,
        ),
        (
            "18.4.dockmodel.analyze_multiple",
            _test_analyze_multiple_dock_models,
        ),
        (
            "18.4.dockmodel.empty_source",
            _test_dock_model_empty_source,
        ),
        (
            "18.4.statistics.interaction_count",
            _test_statistics_interaction_count,
        ),
        (
            "18.4.statistics.total_score",
            _test_statistics_total_score,
        ),
        (
            "18.4.statistics.distance_extrema",
            _test_statistics_distance_extrema,
        ),
        (
            "18.4.statistics.strength_distribution",
            _test_statistics_strength_distribution,
        ),
        (
            "18.4.statistics.in_place",
            _test_statistics_in_place_update,
        ),
        (
            "18.4.statistics.empty_result",
            _test_empty_result_statistics,
        ),
        (
            "18.4.summary.compact_structure",
            _test_compact_summary_structure,
        ),
        (
            "18.4.summary.compact_values",
            _test_compact_summary_values,
        ),
        (
            "18.4.summary.text_nonempty",
            _test_text_summary_nonempty,
        ),
        (
            "18.4.summary.empty_result",
            _test_empty_result_text_summary,
        ),
        (
            "18.4.multipose.ranking_count",
            _test_pose_ranking_count,
        ),
        (
            "18.4.multipose.ranking_order",
            _test_pose_ranking_order,
        ),
        (
            "18.4.multipose.unique_ranks",
            _test_pose_ranking_unique_ranks,
        ),
        (
            "18.4.multipose.persistence_nonempty",
            _test_persistence_records_nonempty,
        ),
        (
            "18.4.multipose.persistence_range",
            _test_persistence_fraction_range,
        ),
        (
            "18.4.multipose.full_persistence",
            _test_fully_persistent_interaction,
        ),
        (
            "18.4.multipose.consensus_nonempty",
            _test_consensus_records_nonempty,
        ),
        (
            "18.4.multipose.consensus_json",
            _test_consensus_is_json_safe,
        ),
        (
            "18.4.multipose.complete_pipeline",
            _test_complete_multipose_analysis,
        ),
        (
            "18.4.multipose.empty",
            _test_empty_multipose_analysis,
        ),
        (
            "18.4.multipose.pose_ids",
            _test_multipose_results_have_pose_ids,
        ),
        (
            "18.4.serialization.charged_atom",
            _test_charged_atom_serialization,
        ),
        (
            "18.4.serialization.charged_group",
            _test_charged_group_serialization,
        ),
        (
            "18.4.serialization.geometry",
            _test_geometry_serialization,
        ),
        (
            "18.4.serialization.interaction",
            _test_interaction_serialization,
        ),
        (
            "18.4.serialization.compact_interaction",
            _test_compact_interaction_serialization,
        ),
        (
            "18.4.serialization.result",
            _test_result_serialization,
        ),
        (
            "18.4.json.result",
            _test_result_json_serialization,
        ),
        (
            "18.4.json.nonfinite_values",
            _test_result_json_rejects_nonfinite_values,
        ),
        (
            "18.4.json.nested_data",
            _test_make_json_safe_nested_data,
        ),
        (
            "18.4.json.interactions",
            _test_interactions_json_serialization,
        ),
        (
            "18.4.tables.interactions",
            _test_interaction_rows,
        ),
        (
            "18.4.tables.groups",
            _test_group_rows,
        ),
        (
            "18.4.tables.statistics",
            _test_statistics_rows,
        ),
        (
            "18.4.tables.residue_summary",
            _test_residue_summary_rows,
        ),
        (
            "18.4.tables.pose_summary",
            _test_pose_summary_rows,
        ),
        (
            "18.4.export.single_payload",
            _test_single_result_export_payload,
        ),
        (
            "18.4.export.dict",
            _test_prepare_export_dict,
        ),
        (
            "18.4.export.json",
            _test_prepare_export_json,
        ),
        (
            "18.4.export.rows",
            _test_prepare_export_rows,
        ),
        (
            "18.4.export.invalid_format",
            _test_prepare_export_invalid_format,
        ),
        (
            "18.4.multipose_serialization.dict",
            _test_multipose_dictionary_serialization,
        ),
        (
            "18.4.multipose_serialization.json",
            _test_multipose_json_serialization,
        ),
        (
            "18.4.multipose_serialization.payload",
            _test_multipose_export_payload,
        ),
        (
            "18.4.chimerax.atom_spec",
            _test_atom_chimerax_spec,
        ),
        (
            "18.4.chimerax.residue_spec",
            _test_residue_chimerax_spec,
        ),
        (
            "18.4.chimerax.interaction_spec",
            _test_interaction_chimerax_spec,
        ),
        (
            "18.4.chimerax.selection_command",
            _test_result_selection_command,
        ),
        (
            "18.4.chimerax.pseudobond_command",
            _test_pseudobond_command_generation,
        ),
        (
            "18.4.chimerax.visualization_commands",
            _test_visualization_command_generation,
        ),
        (
            "18.4.chimerax.export_record",
            _test_chimerax_record_serialization,
        ),
        (
            "18.4.chimerax.no_runtime_required",
            _test_chimerax_generation_without_runtime,
        ),
        (
            "18.4.round_trip.core_fields",
            _test_result_json_round_trip_core_fields,
        ),
        (
            "18.4.round_trip.compact_full_count",
            _test_compact_and_full_serialization_counts_match,
        ),
        (
            "18.4.round_trip.no_raw_atoms",
            _test_serialization_does_not_include_raw_atom_objects,
        ),
    ]


# =============================================================================
# 18.4.19. SECTION RUNNER
# =============================================================================


def run_salt_bridge_integration_serialization_tests(
    *,
    report: Optional[
        _SelfTestReport
    ] = None,
    raise_on_failure: bool = False,
    print_report: bool = False,
) -> _SelfTestReport:
    """
    Run all Section 18.4 integration and serialization self-tests.

    Parameters
    ----------
    report
        Optional existing self-test report.
    raise_on_failure
        Whether the first failure should raise an exception.
    print_report
        Whether the updated report should be printed.

    Returns
    -------
    _SelfTestReport
        Updated self-test report.
    """

    resolved_report = (
        _run_self_test_group(
            get_salt_bridge_integration_serialization_tests(),
            report=report,
            raise_on_failure=(
                raise_on_failure
            ),
        )
    )

    if print_report:
        _print_self_test_report(
            resolved_report
        )

    return resolved_report


# =============================================================================
# 18.5. FINAL SELF-TEST RUNNER
# =============================================================================


# =============================================================================
# 18.5.1. DATE AND ENVIRONMENT UTILITIES
# =============================================================================


def _self_test_timestamp() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.

    Returns
    -------
    str
        Current UTC timestamp.
    """

    from datetime import (
        datetime,
        timezone,
    )

    return datetime.now(
        timezone.utc
    ).isoformat()


def _collect_self_test_environment() -> Dict[str, Any]:
    """
    Collect basic runtime information for the self-test report.

    Returns
    -------
    Dict[str, Any]
        JSON-safe runtime metadata.
    """

    import platform
    import sys

    return {
        "python_version": (
            sys.version.split()[0]
        ),
        "python_implementation": (
            platform.python_implementation()
        ),
        "platform": (
            platform.platform()
        ),
        "machine": (
            platform.machine()
        ),
        "processor": (
            platform.processor()
        ),
        "module_name": __name__,
        "module_version": __version__,
        "chimerax_available": bool(
            HAS_CHIMERAX
        ),
    }


# =============================================================================
# 18.5.2. SECTION REGISTRY
# =============================================================================


def get_salt_bridge_self_test_sections(
) -> List[
    Tuple[
        str,
        Callable[
            ...,
            _SelfTestReport,
        ],
    ]
]:
    """
    Return the ordered salt-bridge self-test sections.

    Returns
    -------
    List[Tuple[str, Callable[..., _SelfTestReport]]]
        Ordered section runner definitions.
    """

    return [
        (
            "18.1.test_infrastructure",
            run_salt_bridge_test_infrastructure,
        ),
        (
            "18.2.recognition_and_geometry",
            run_salt_bridge_recognition_geometry_tests,
        ),
        (
            "18.3.detection_and_classification",
            run_salt_bridge_detection_classification_tests,
        ),
        (
            "18.4.integration_and_serialization",
            run_salt_bridge_integration_serialization_tests,
        ),
    ]


def get_salt_bridge_self_test_names(
) -> List[str]:
    """
    Return all registered salt-bridge self-test names.

    Returns
    -------
    List[str]
        Registered test names in execution order.
    """

    test_names = [
        "18.1.test_infrastructure_smoke_test"
    ]

    test_names.extend(
        test_name
        for test_name, _
        in get_salt_bridge_recognition_geometry_tests()
    )

    test_names.extend(
        test_name
        for test_name, _
        in get_salt_bridge_detection_classification_tests()
    )

    test_names.extend(
        test_name
        for test_name, _
        in get_salt_bridge_integration_serialization_tests()
    )

    return test_names


# =============================================================================
# 18.5.3. SECTION 18.1 RUNNER ADAPTER
# =============================================================================


def run_salt_bridge_test_infrastructure(
    *,
    report: Optional[
        _SelfTestReport
    ] = None,
    raise_on_failure: bool = False,
    print_report: bool = False,
) -> _SelfTestReport:
    """
    Run the Section 18.1 infrastructure self-test.

    This adapter gives Section 18.1 the same interface used by the
    remaining section runners.

    Parameters
    ----------
    report
        Optional existing self-test report.
    raise_on_failure
        Whether a failure should immediately raise an exception.
    print_report
        Whether the updated report should be printed.

    Returns
    -------
    _SelfTestReport
        Updated self-test report.
    """

    resolved_report = (
        report
        if report is not None
        else _SelfTestReport()
    )

    test_record = (
        run_salt_bridge_test_infrastructure_smoke_test(
            raise_on_failure=(
                raise_on_failure
            )
        )
    )

    resolved_report.add_record(
        test_record
    )

    if print_report:
        _print_self_test_report(
            resolved_report
        )

    return resolved_report


# =============================================================================
# 18.5.4. REPORT VALIDATION
# =============================================================================


def _validate_self_test_registry() -> None:
    """
    Validate the final self-test registry.

    Raises
    ------
    SaltBridgeSelfTestError
        If duplicate or invalid test names are found.
    """

    test_names = (
        get_salt_bridge_self_test_names()
    )

    if not test_names:
        raise SaltBridgeSelfTestError(
            "The salt-bridge self-test registry is empty."
        )

    normalized_names = [
        str(
            test_name
        ).strip()
        for test_name in test_names
    ]

    if any(
        not test_name
        for test_name in normalized_names
    ):
        raise SaltBridgeSelfTestError(
            "Self-test names must not be empty."
        )

    duplicate_names = sorted(
        {
            test_name
            for test_name
            in normalized_names
            if normalized_names.count(
                test_name
            ) > 1
        }
    )

    if duplicate_names:
        raise SaltBridgeSelfTestError(
            "Duplicate self-test names were found: "
            + ", ".join(
                duplicate_names
            )
        )


def _validate_final_self_test_report(
    report: _SelfTestReport,
) -> None:
    """
    Validate invariants of a completed self-test report.

    Parameters
    ----------
    report
        Completed self-test report.

    Raises
    ------
    SaltBridgeSelfTestError
        If the report is internally inconsistent.
    """

    _assert_is_instance(
        report,
        _SelfTestReport,
    )

    _assert_equal(
        report.test_count,
        (
            report.passed_count
            + report.failed_count
        ),
        (
            "Self-test report counters "
            "are inconsistent."
        ),
    )

    registered_test_count = len(
        get_salt_bridge_self_test_names()
    )

    _assert_equal(
        report.test_count,
        registered_test_count,
        (
            "Executed test count does not match "
            "the registered test count."
        ),
    )

    record_names = [
        record.name
        for record in report.records
    ]

    _assert_equal(
        len(
            record_names
        ),
        len(
            set(
                record_names
            )
        ),
        (
            "The final self-test report contains "
            "duplicate records."
        ),
    )

    for record in report.records:
        _assert_true(
            record.duration_seconds >= 0.0,
            (
                "Self-test duration cannot "
                "be negative."
            ),
        )

        if record.passed:
            _assert_is_none(
                record.exception_type,
                (
                    "Passed tests must not contain "
                    "an exception type."
                ),
            )

        else:
            _assert_true(
                bool(
                    record.message
                ),
                (
                    "Failed tests must contain "
                    "an explanatory message."
                ),
            )


# =============================================================================
# 18.5.5. FINAL REPORT FORMATTING
# =============================================================================


def _format_self_test_section_summary(
    report: _SelfTestReport,
) -> List[str]:
    """
    Build section-level summary lines.

    Parameters
    ----------
    report
        Completed self-test report.

    Returns
    -------
    List[str]
        Formatted section summary lines.
    """

    section_prefixes = (
        (
            "18.1",
            "Infrastructure",
        ),
        (
            "18.2",
            "Recognition and geometry",
        ),
        (
            "18.3",
            "Detection and classification",
        ),
        (
            "18.4",
            "Integration and serialization",
        ),
    )

    summary_lines: List[str] = []

    for prefix, label in section_prefixes:
        section_records = [
            record
            for record in report.records
            if record.name.startswith(
                prefix
            )
        ]

        passed_count = sum(
            1
            for record in section_records
            if record.passed
        )

        failed_count = (
            len(
                section_records
            )
            - passed_count
        )

        status = (
            "PASS"
            if failed_count == 0
            and section_records
            else "FAIL"
        )

        summary_lines.append(
            (
                f"[{status}] {prefix} "
                f"{label}: "
                f"{passed_count}/"
                f"{len(section_records)} passed"
            )
        )

    return summary_lines


def format_salt_bridge_self_test_report(
    report: _SelfTestReport,
    *,
    include_section_summary: bool = True,
    include_records: bool = True,
    include_environment: bool = False,
) -> str:
    """
    Format the final salt-bridge self-test report.

    Parameters
    ----------
    report
        Completed self-test report.
    include_section_summary
        Whether section-level results should be included.
    include_records
        Whether individual test records should be included.
    include_environment
        Whether runtime metadata should be included.

    Returns
    -------
    str
        Formatted report.
    """

    total_duration = sum(
        record.duration_seconds
        for record in report.records
    )

    status = (
        "SUCCESS"
        if report.success
        else "FAILURE"
    )

    lines = [
        "=" * 79,
        "DockAnalyzer saltbridge.py self-test report",
        "=" * 79,
        f"Module: {report.module_name}",
        (
            "Version: "
            f"{report.module_version}"
        ),
        (
            "Started: "
            f"{report.started_at or 'unknown'}"
        ),
        (
            "Finished: "
            f"{report.finished_at or 'unknown'}"
        ),
        (
            "Duration: "
            f"{total_duration:.6f} s"
        ),
        (
            "Registered tests: "
            f"{report.test_count}"
        ),
        (
            "Passed: "
            f"{report.passed_count}"
        ),
        (
            "Failed: "
            f"{report.failed_count}"
        ),
        (
            "Final status: "
            f"{status}"
        ),
    ]

    if include_section_summary:
        lines.extend(
            [
                "",
                "-" * 79,
                "Section summary",
                "-" * 79,
            ]
        )

        lines.extend(
            _format_self_test_section_summary(
                report
            )
        )

    if include_environment:
        environment = (
            report.metadata.get(
                "environment",
                {},
            )
        )

        lines.extend(
            [
                "",
                "-" * 79,
                "Environment",
                "-" * 79,
            ]
        )

        if isinstance(
            environment,
            Mapping,
        ):
            for key in sorted(
                environment
            ):
                lines.append(
                    f"{key}: "
                    f"{environment[key]}"
                )

    if include_records:
        lines.extend(
            [
                "",
                "-" * 79,
                "Individual tests",
                "-" * 79,
            ]
        )

        lines.extend(
            _format_self_test_record(
                record
            )
            for record in report.records
        )

    if not report.success:
        failed_records = [
            record
            for record in report.records
            if not record.passed
        ]

        lines.extend(
            [
                "",
                "-" * 79,
                "Failures",
                "-" * 79,
            ]
        )

        for failed_record in (
            failed_records
        ):
            exception_name = (
                failed_record.exception_type
                or "UnknownError"
            )

            lines.append(
                (
                    f"{failed_record.name}: "
                    f"{exception_name}: "
                    f"{failed_record.message}"
                )
            )

    lines.append(
        "=" * 79
    )

    return "\n".join(
        lines
    )


def print_salt_bridge_self_test_report(
    report: _SelfTestReport,
    *,
    include_section_summary: bool = True,
    include_records: bool = True,
    include_environment: bool = False,
) -> None:
    """
    Print the final salt-bridge self-test report.

    Parameters
    ----------
    report
        Completed self-test report.
    include_section_summary
        Whether section-level results should be shown.
    include_records
        Whether individual test records should be shown.
    include_environment
        Whether runtime metadata should be shown.
    """

    print(
        format_salt_bridge_self_test_report(
            report,
            include_section_summary=(
                include_section_summary
            ),
            include_records=(
                include_records
            ),
            include_environment=(
                include_environment
            ),
        )
    )


# =============================================================================
# 18.5.6. REPORT SERIALIZATION
# =============================================================================


def salt_bridge_self_test_report_to_dict(
    report: _SelfTestReport,
) -> Dict[str, Any]:
    """
    Convert the final self-test report into a JSON-safe dictionary.

    Parameters
    ----------
    report
        Completed self-test report.

    Returns
    -------
    Dict[str, Any]
        JSON-safe report mapping.
    """

    report_dict = report.to_dict()

    report_dict[
        "schema"
    ] = (
        "dockanalyzer.saltbridge."
        "self_tests"
    )

    report_dict[
        "schema_version"
    ] = "1.0"

    report_dict[
        "total_duration_seconds"
    ] = sum(
        record.duration_seconds
        for record in report.records
    )

    report_dict[
        "registered_test_names"
    ] = (
        get_salt_bridge_self_test_names()
    )

    return make_json_safe(
        report_dict
    )


def serialize_salt_bridge_self_test_report(
    report: _SelfTestReport,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
) -> str:
    """
    Serialize the final self-test report as strict JSON.

    Parameters
    ----------
    report
        Completed self-test report.
    indent
        JSON indentation.
    sort_keys
        Whether dictionary keys should be sorted.

    Returns
    -------
    str
        JSON document.
    """

    return json.dumps(
        salt_bridge_self_test_report_to_dict(
            report
        ),
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


# =============================================================================
# 18.5.7. FINAL TEST EXECUTION
# =============================================================================


def run_self_tests(
    *,
    raise_on_failure: bool = False,
    print_report: bool = True,
    include_section_summary: bool = True,
    include_records: bool = True,
    include_environment: bool = False,
    validate_registry: bool = True,
    validate_report: bool = True,
) -> _SelfTestReport:
    """
    Run the complete saltbridge.py self-test suite.

    The runner executes Sections 18.1 through 18.4 in order.

    Parameters
    ----------
    raise_on_failure
        Whether execution should stop at the first failed test.
    print_report
        Whether the final report should be printed.
    include_section_summary
        Whether the printed report should include section summaries.
    include_records
        Whether the printed report should include individual test records.
    include_environment
        Whether the printed report should include runtime information.
    validate_registry
        Whether the self-test registry should be validated before execution.
    validate_report
        Whether the final report should be validated after execution.

    Returns
    -------
    _SelfTestReport
        Completed self-test report.

    Raises
    ------
    SaltBridgeSelfTestError
        If registry validation fails, a test fails while
        ``raise_on_failure`` is true, or final report validation fails.
    """

    if validate_registry:
        _validate_self_test_registry()

    report = _SelfTestReport(
        module_name="saltbridge",
        module_version=__version__,
        records=[],
        started_at=(
            _self_test_timestamp()
        ),
        finished_at=None,
        metadata={
            "environment": (
                _collect_self_test_environment()
            ),
            "section_count": 4,
            "registered_test_count": len(
                get_salt_bridge_self_test_names()
            ),
            "raise_on_failure": bool(
                raise_on_failure
            ),
        },
    )

    section_results: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for (
        section_name,
        section_runner,
    ) in get_salt_bridge_self_test_sections():
        initial_record_count = (
            report.test_count
        )

        report = section_runner(
            report=report,
            raise_on_failure=(
                raise_on_failure
            ),
            print_report=False,
        )

        new_records = report.records[
            initial_record_count:
        ]

        section_results[
            section_name
        ] = {
            "test_count": len(
                new_records
            ),
            "passed_count": sum(
                1
                for record in new_records
                if record.passed
            ),
            "failed_count": sum(
                1
                for record in new_records
                if not record.passed
            ),
            "duration_seconds": sum(
                record.duration_seconds
                for record in new_records
            ),
        }

    report.finished_at = (
        _self_test_timestamp()
    )

    report.metadata[
        "sections"
    ] = section_results

    report.metadata[
        "success"
    ] = report.success

    if validate_report:
        _validate_final_self_test_report(
            report
        )

    if print_report:
        print_salt_bridge_self_test_report(
            report,
            include_section_summary=(
                include_section_summary
            ),
            include_records=(
                include_records
            ),
            include_environment=(
                include_environment
            ),
        )

    if (
        raise_on_failure
        and not report.success
    ):
        failed_test_names = [
            record.name
            for record in report.records
            if not record.passed
        ]

        raise SaltBridgeSelfTestError(
            "Salt-bridge self-tests failed: "
            + ", ".join(
                failed_test_names
            )
        )

    return report


def run_salt_bridge_self_tests(
    **options: Any,
) -> _SelfTestReport:
    """
    Alias for :func:`run_self_tests`.

    Parameters
    ----------
    **options
        Options forwarded to ``run_self_tests``.

    Returns
    -------
    _SelfTestReport
        Completed self-test report.
    """

    return run_self_tests(
        **options
    )


# =============================================================================
# 18.5.8. COMMAND-LINE ENTRY POINT
# =============================================================================


def _self_test_exit_code(
    report: _SelfTestReport,
) -> int:
    """
    Return a process exit code for a self-test report.

    Parameters
    ----------
    report
        Completed self-test report.

    Returns
    -------
    int
        Zero for success and one for failure.
    """

    return (
        0
        if report.success
        else 1
    )


def _run_self_tests_from_command_line() -> int:
    """
    Run self-tests from the command line.

    Returns
    -------
    int
        Process exit code.
    """

    import argparse

    parser = argparse.ArgumentParser(
        prog="saltbridge.py",
        description=(
            "Run DockAnalyzer salt-bridge "
            "module self-tests."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Print only the section summary "
            "and final status."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop execution at the first "
            "failed test."
        ),
    )

    parser.add_argument(
        "--environment",
        action="store_true",
        help=(
            "Include runtime environment "
            "information in the report."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help=(
            "Print the final report as JSON."
        ),
    )

    arguments = parser.parse_args()

    try:
        report = run_self_tests(
            raise_on_failure=(
                arguments.fail_fast
            ),
            print_report=False,
        )

    except SaltBridgeSelfTestError as error:
        print(
            (
                "Salt-bridge self-test "
                f"runner error: {error}"
            )
        )

        return 1

    if arguments.json_output:
        print(
            serialize_salt_bridge_self_test_report(
                report
            )
        )

    else:
        print_salt_bridge_self_test_report(
            report,
            include_section_summary=True,
            include_records=(
                not arguments.quiet
            ),
            include_environment=(
                arguments.environment
            ),
        )

    return _self_test_exit_code(
        report
    )


if __name__ == "__main__":
    raise SystemExit(
        _run_self_tests_from_command_line()
    )






