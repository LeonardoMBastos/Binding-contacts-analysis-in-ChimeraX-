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








