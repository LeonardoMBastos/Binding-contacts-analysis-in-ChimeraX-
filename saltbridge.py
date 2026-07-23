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











