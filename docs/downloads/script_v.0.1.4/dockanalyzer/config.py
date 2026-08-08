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
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))

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
    return True


if __name__ == "__main__":
    _self_test()
    create_output_directories()
    print_configuration()
