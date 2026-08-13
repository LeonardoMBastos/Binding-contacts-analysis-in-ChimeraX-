"""Configuration defaults for DockAnalyzer.

The module is deliberately side-effect free: importing it does not create
directories, open files, or require ChimeraX.  Distances are expressed in
angstroms, angles in degrees, and energies in kcal/mol.
"""

from pathlib import Path
from ._version import __version__


# General settings
PROJECT_NAME = "DockAnalyzer"
VERSION = __version__
AUTHOR = "Leonardo Bastos"
VERBOSE = True
SAVE_LOG = True
OVERWRITE_OUTPUT = True


# Input model conventions
RECEPTOR_MODEL_ID = 1
FIRST_POSE_MODEL_ID = 2
RECEPTOR_NAME = "Receptor"
LIGAND_NAME = "Ligand"


# Output paths.  These are declarations only; importing this module does not
# create them.  Relative paths are resolved when an output operation runs.
OUTPUT_DIR = Path("DockAnalyzer_Output")
CSV_DIR = OUTPUT_DIR / "CSV"
EXCEL_DIR = OUTPUT_DIR / "Excel"
IMAGE_DIR = OUTPUT_DIR / "Images"
SESSION_DIR = OUTPUT_DIR / "Sessions"
REPORT_DIR = OUTPUT_DIR / "Reports"
JSON_DIR = OUTPUT_DIR / "JSON"
LOG_DIR = OUTPUT_DIR / "Logs"

DIRECTORIES = (
    OUTPUT_DIR,
    CSV_DIR,
    EXCEL_DIR,
    IMAGE_DIR,
    SESSION_DIR,
    REPORT_DIR,
    JSON_DIR,
    LOG_DIR,
)


# Contact analysis
CONTACT_DISTANCE = 4.0
MAX_CONTACT_DISTANCE = 5.0
MIN_CONTACT_DISTANCE = 2.0


# Hydrogen bonds
HBOND_MAX_DISTANCE = 3.5
HBOND_MIN_ANGLE = 120.0
HBOND_SHOW_DASHES = True


# Hydrophobic interactions
HYDROPHOBIC_DISTANCE = 4.5
HYDROPHOBIC_RESIDUES = {
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


# Pi interactions
PI_STACK_MAX_DISTANCE = 5.5
PI_STACK_MAX_ANGLE = 30.0
PI_CATION_MAX_DISTANCE = 6.0
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
CATIONIC_RESIDUES = {"ARG", "LYS", "HIS"}


# Salt bridges
SALT_BRIDGE_DISTANCE = 4.0
POSITIVE_RESIDUES = {"ARG", "LYS", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}


# Clash analysis
CLASH_OVERLAP = 0.4


# Visualization
BACKGROUND_COLOR = "white"
PROTEIN_COLOR = "light gray"
LIGAND_COLOR = "gold"
HBOND_COLOR = "green"
HYDROPHOBIC_COLOR = "orange"
PI_STACK_COLOR = "blue"
PI_CATION_COLOR = "purple"
SALT_BRIDGE_COLOR = "red"
CONTACT_COLOR = "cyan"
IMAGE_WIDTH = 2400
IMAGE_HEIGHT = 1800
IMAGE_DPI = 300


# Performance plots
GENERATE_PERFORMANCE_PLOTS = True
PERFORMANCE_TIMELINE_FILENAME = "execution_timeline.png"
PERFORMANCE_RUNTIME_FILENAME = "stage_runtime.png"
PERFORMANCE_IMAGE_WIDTH = 1800
PERFORMANCE_INCLUDE_GLOBAL_STAGES = True
PERFORMANCE_INCLUDE_ZERO_DURATION = False


# Export options
EXPORT_CSV = True
EXPORT_EXCEL = False
EXPORT_JSON = True
EXPORT_IMAGES = False
EXPORT_SESSION = False


# Scoring weights
SCORE = {
    "hydrogen_bond": 2,
    "hydrophobic": 1,
    "pi_stack": 2,
    "pi_cation": 2,
    "salt_bridge": 3,
    "clash": -2,
}


# Score v2 Stage 4: ligand size/opportunity normalization
#
# These values mirror the conservative defaults validated in
# scoring_scorev2_stage1-4.py. Affinity and RMSD are intentionally absent:
# they remain external validation variables and never enter the score.
SCORING_OPPORTUNITY_NORMALIZATION_ENABLED = True
SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS = 8.0
SCORING_OPPORTUNITY_SIZE_EXPONENT = 0.20
SCORING_OPPORTUNITY_SIZE_MIN_FACTOR = 0.70
SCORING_OPPORTUNITY_FAMILY_MIN_FACTOR = 0.65

SCORING_OPPORTUNITY_REFERENCE_COUNTS = {
    "contact": 8.0,
    "hydrophobic": 4.0,
    "hydrogen_bond": 3.0,
    "pi": 1.0,
    "salt_bridge": 1.0,
}

SCORING_OPPORTUNITY_FAMILY_EXPONENTS = {
    "contact": 0.10,
    "hydrophobic": 0.20,
    "hydrogen_bond": 0.15,
    "pi": 0.15,
    "salt_bridge": 0.15,
}

# Ready-to-consume Stage 4 settings block. Keeping the flat constants above
# makes individual parameters easy to inspect and override, while this mapping
# provides one reproducible configuration object for the scoring integration.
SCORING_OPPORTUNITY_NORMALIZATION = {
    "enabled": SCORING_OPPORTUNITY_NORMALIZATION_ENABLED,
    "size_reference_heavy_atoms": (
        SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS
    ),
    "size_exponent": SCORING_OPPORTUNITY_SIZE_EXPONENT,
    "size_min_factor": SCORING_OPPORTUNITY_SIZE_MIN_FACTOR,
    "family_min_factor": SCORING_OPPORTUNITY_FAMILY_MIN_FACTOR,
    "reference_counts": dict(SCORING_OPPORTUNITY_REFERENCE_COUNTS),
    "family_exponents": dict(SCORING_OPPORTUNITY_FAMILY_EXPONENTS),
}


# Supported input files
SUPPORTED_EXTENSIONS = {".pdb", ".mol2", ".sdf"}


def validate_configuration():
    """Validate configuration invariants and return ``True``.

    Raises:
        ValueError: If a cutoff, angle, image dimension, model identifier, or
            supported file extension is invalid.
    """

    positive_distances = {
        "CONTACT_DISTANCE": CONTACT_DISTANCE,
        "MAX_CONTACT_DISTANCE": MAX_CONTACT_DISTANCE,
        "MIN_CONTACT_DISTANCE": MIN_CONTACT_DISTANCE,
        "HBOND_MAX_DISTANCE": HBOND_MAX_DISTANCE,
        "HYDROPHOBIC_DISTANCE": HYDROPHOBIC_DISTANCE,
        "PI_STACK_MAX_DISTANCE": PI_STACK_MAX_DISTANCE,
        "PI_CATION_MAX_DISTANCE": PI_CATION_MAX_DISTANCE,
        "SALT_BRIDGE_DISTANCE": SALT_BRIDGE_DISTANCE,
        "CLASH_OVERLAP": CLASH_OVERLAP,
    }
    for name, value in positive_distances.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive number".format(name))

    if not MIN_CONTACT_DISTANCE <= CONTACT_DISTANCE <= MAX_CONTACT_DISTANCE:
        raise ValueError(
            "contact distances must satisfy "
            "MIN_CONTACT_DISTANCE <= CONTACT_DISTANCE <= MAX_CONTACT_DISTANCE"
        )

    for name, value in {
        "HBOND_MIN_ANGLE": HBOND_MIN_ANGLE,
        "PI_STACK_MAX_ANGLE": PI_STACK_MAX_ANGLE,
    }.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("{} must be numeric".format(name))
        if not 0.0 <= value <= 180.0:
            raise ValueError("{} must be between 0 and 180 degrees".format(name))

    if RECEPTOR_MODEL_ID < 1 or FIRST_POSE_MODEL_ID <= RECEPTOR_MODEL_ID:
        raise ValueError("pose model identifiers must follow the receptor model")

    for name, value in {
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "IMAGE_DPI": IMAGE_DPI,
        "PERFORMANCE_IMAGE_WIDTH": PERFORMANCE_IMAGE_WIDTH,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))

    for name, value in {
        "GENERATE_PERFORMANCE_PLOTS": GENERATE_PERFORMANCE_PLOTS,
        "PERFORMANCE_INCLUDE_GLOBAL_STAGES": PERFORMANCE_INCLUDE_GLOBAL_STAGES,
        "PERFORMANCE_INCLUDE_ZERO_DURATION": PERFORMANCE_INCLUDE_ZERO_DURATION,
    }.items():
        if not isinstance(value, bool):
            raise ValueError("{} must be a boolean".format(name))

    for name, value in {
        "PERFORMANCE_TIMELINE_FILENAME": PERFORMANCE_TIMELINE_FILENAME,
        "PERFORMANCE_RUNTIME_FILENAME": PERFORMANCE_RUNTIME_FILENAME,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be a non-empty filename".format(name))
        if Path(value).name != value:
            raise ValueError("{} must be a filename without directories".format(name))
        if Path(value).suffix.lower() != ".png":
            raise ValueError("{} must use the .png extension".format(name))

    if not isinstance(SCORING_OPPORTUNITY_NORMALIZATION_ENABLED, bool):
        raise ValueError(
            "SCORING_OPPORTUNITY_NORMALIZATION_ENABLED must be a boolean"
        )

    positive_opportunity_values = {
        "SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS": (
            SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS
        ),
    }
    for name, value in positive_opportunity_values.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0.0
        ):
            raise ValueError("{} must be a positive number".format(name))

    unit_interval_values = {
        "SCORING_OPPORTUNITY_SIZE_MIN_FACTOR": (
            SCORING_OPPORTUNITY_SIZE_MIN_FACTOR
        ),
        "SCORING_OPPORTUNITY_FAMILY_MIN_FACTOR": (
            SCORING_OPPORTUNITY_FAMILY_MIN_FACTOR
        ),
    }
    for name, value in unit_interval_values.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("{} must be between 0 and 1".format(name))

    nonnegative_exponents = {
        "SCORING_OPPORTUNITY_SIZE_EXPONENT": (
            SCORING_OPPORTUNITY_SIZE_EXPONENT
        ),
        **{
            "SCORING_OPPORTUNITY_FAMILY_EXPONENTS[{}]".format(family): value
            for family, value in SCORING_OPPORTUNITY_FAMILY_EXPONENTS.items()
        },
    }
    for name, value in nonnegative_exponents.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0.0
        ):
            raise ValueError("{} must be a non-negative number".format(name))

    opportunity_families = {
        "contact",
        "hydrophobic",
        "hydrogen_bond",
        "pi",
        "salt_bridge",
    }
    if set(SCORING_OPPORTUNITY_REFERENCE_COUNTS) != opportunity_families:
        raise ValueError(
            "SCORING_OPPORTUNITY_REFERENCE_COUNTS must define every "
            "Score v2 opportunity family"
        )
    if set(SCORING_OPPORTUNITY_FAMILY_EXPONENTS) != opportunity_families:
        raise ValueError(
            "SCORING_OPPORTUNITY_FAMILY_EXPONENTS must define every "
            "Score v2 opportunity family"
        )
    for family, value in SCORING_OPPORTUNITY_REFERENCE_COUNTS.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0.0
        ):
            raise ValueError(
                "SCORING_OPPORTUNITY_REFERENCE_COUNTS[{}] must be a "
                "positive number".format(family)
            )

    expected_opportunity_settings = {
        "enabled",
        "size_reference_heavy_atoms",
        "size_exponent",
        "size_min_factor",
        "family_min_factor",
        "reference_counts",
        "family_exponents",
    }
    if set(SCORING_OPPORTUNITY_NORMALIZATION) != expected_opportunity_settings:
        raise ValueError(
            "SCORING_OPPORTUNITY_NORMALIZATION contains an invalid key set"
        )

    if any(not extension.startswith(".") for extension in SUPPORTED_EXTENSIONS):
        raise ValueError("supported file extensions must start with a period")

    return True


def create_output_directories():
    """Create the declared output directories explicitly.

    This function preserves the original public API and is never called
    automatically during module import.
    """

    for folder in DIRECTORIES:
        folder.mkdir(parents=True, exist_ok=True)


def print_configuration():
    """Print a concise summary of the effective configuration."""

    print("=" * 70)
    print(PROJECT_NAME)
    print("Version:", VERSION)
    print("=" * 70)
    print("Output folder : {}".format(OUTPUT_DIR))
    print("Contact cutoff: {:.1f} Å".format(CONTACT_DISTANCE))
    print("HBond cutoff  : {:.1f} Å".format(HBOND_MAX_DISTANCE))
    print("Hydrophobic   : {:.1f} Å".format(HYDROPHOBIC_DISTANCE))
    print("π-π cutoff    : {:.1f} Å".format(PI_STACK_MAX_DISTANCE))
    print("π-cation      : {:.1f} Å".format(PI_CATION_MAX_DISTANCE))
    print("Performance   : {}".format("enabled" if GENERATE_PERFORMANCE_PLOTS else "disabled"))
    print("Performance dir: {}".format(IMAGE_DIR))
    print(
        "Score v2 opp. : {}".format(
            "enabled"
            if SCORING_OPPORTUNITY_NORMALIZATION_ENABLED
            else "disabled"
        )
    )
    print(
        "Score v2 size : ref={:.1f} heavy atoms, exponent={:.2f}".format(
            SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS,
            SCORING_OPPORTUNITY_SIZE_EXPONENT,
        )
    )
    print("=" * 70)


def _self_test():
    """Run deterministic checks that do not touch the file system."""

    assert validate_configuration() is True
    assert OUTPUT_DIR not in OUTPUT_DIR.parents
    assert all(isinstance(folder, Path) for folder in DIRECTORIES)
    assert set(SCORE) == {
        "hydrogen_bond",
        "hydrophobic",
        "pi_stack",
        "pi_cation",
        "salt_bridge",
        "clash",
    }
    assert isinstance(GENERATE_PERFORMANCE_PLOTS, bool)
    assert PERFORMANCE_TIMELINE_FILENAME.endswith(".png")
    assert PERFORMANCE_RUNTIME_FILENAME.endswith(".png")
    assert PERFORMANCE_IMAGE_WIDTH > 0
    assert SCORING_OPPORTUNITY_NORMALIZATION_ENABLED is True
    assert SCORING_OPPORTUNITY_SIZE_REFERENCE_HEAVY_ATOMS == 8.0
    assert SCORING_OPPORTUNITY_SIZE_EXPONENT == 0.20
    assert SCORING_OPPORTUNITY_SIZE_MIN_FACTOR == 0.70
    assert SCORING_OPPORTUNITY_FAMILY_MIN_FACTOR == 0.65
    assert SCORING_OPPORTUNITY_REFERENCE_COUNTS["hydrophobic"] == 4.0
    assert SCORING_OPPORTUNITY_FAMILY_EXPONENTS["contact"] == 0.10
    assert "affinity" not in SCORING_OPPORTUNITY_NORMALIZATION
    assert "rmsd" not in SCORING_OPPORTUNITY_NORMALIZATION
    return True


if __name__ == "__main__":
    _self_test()
    create_output_directories()
    print_configuration()
