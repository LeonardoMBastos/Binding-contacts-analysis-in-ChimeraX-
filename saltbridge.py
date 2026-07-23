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






