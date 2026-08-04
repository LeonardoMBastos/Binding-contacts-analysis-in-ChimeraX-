"""
===============================================================================
DockAnalyzer Configuration File
-------------------------------------------------------------------------------
Author : Leonardo Bastos & ChatGPT
Project: DockAnalyzer
Version: 0.1.0

This file contains every configurable parameter used by DockAnalyzer.

Changing values here automatically affects the entire project.

Units
-----
Distances : Angstrom (Å)
Angles    : Degrees
Energy    : kcal/mol
===============================================================================
"""

from pathlib import Path

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

PROJECT_NAME = "DockAnalyzer"

VERSION = "0.1.0"

AUTHOR = "Leonardo Bastos"

VERBOSE = True

SAVE_LOG = True

OVERWRITE_OUTPUT = True


# =============================================================================
# INPUT
# =============================================================================

# The receptor is always opened FIRST in ChimeraX.
# All subsequent models are considered docking poses.

RECEPTOR_MODEL_ID = 1

FIRST_POSE_MODEL_ID = 2

RECEPTOR_NAME = "Receptor"

LIGAND_NAME = "Ligand"


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = Path("DockAnalyzer_Output")

CSV_DIR = OUTPUT_DIR / "CSV"

EXCEL_DIR = OUTPUT_DIR / "Excel"

IMAGE_DIR = OUTPUT_DIR / "Images"

SESSION_DIR = OUTPUT_DIR / "Sessions"

REPORT_DIR = OUTPUT_DIR / "Reports"

JSON_DIR = OUTPUT_DIR / "JSON"

LOG_DIR = OUTPUT_DIR / "Logs"


# =============================================================================
# CONTACT ANALYSIS
# =============================================================================

CONTACT_DISTANCE = 4.0

MAX_CONTACT_DISTANCE = 5.0

MIN_CONTACT_DISTANCE = 2.0


# =============================================================================
# HYDROGEN BONDS
# =============================================================================

HBOND_MAX_DISTANCE = 3.5

HBOND_MIN_ANGLE = 120.0

HBOND_SHOW_DASHES = True


# =============================================================================
# HYDROPHOBIC INTERACTIONS
# =============================================================================

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
    "TYR"

}


# =============================================================================
# PI INTERACTIONS
# =============================================================================

PI_STACK_MAX_DISTANCE = 5.5

PI_STACK_MAX_ANGLE = 30.0

PI_CATION_MAX_DISTANCE = 6.0


AROMATIC_RESIDUES = {

    "PHE",
    "TYR",
    "TRP",
    "HIS"

}


CATIONIC_RESIDUES = {

    "ARG",
    "LYS",
    "HIS"

}


# =============================================================================
# SALT BRIDGES
# =============================================================================

SALT_BRIDGE_DISTANCE = 4.0


POSITIVE_RESIDUES = {

    "ARG",
    "LYS",
    "HIS"

}


NEGATIVE_RESIDUES = {

    "ASP",
    "GLU"

}


# =============================================================================
# CLASH ANALYSIS
# =============================================================================

CLASH_OVERLAP = 0.4


# =============================================================================
# VISUALIZATION
# =============================================================================

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


# =============================================================================
# EXPORT OPTIONS
# =============================================================================

EXPORT_CSV = True

EXPORT_EXCEL = True

EXPORT_JSON = True

EXPORT_IMAGES = True

EXPORT_SESSION = True


# =============================================================================
# SCORING
# =============================================================================

SCORE = {

    "hydrogen_bond": 2,

    "hydrophobic": 1,

    "pi_stack": 2,

    "pi_cation": 2,

    "salt_bridge": 3,

    "clash": -2

}


# =============================================================================
# SUPPORTED FILES
# =============================================================================

SUPPORTED_EXTENSIONS = {

    ".pdb",

    ".mol2",

    ".sdf"

}


# =============================================================================
# CREATE OUTPUT FOLDERS
# =============================================================================

DIRECTORIES = [

    OUTPUT_DIR,

    CSV_DIR,

    EXCEL_DIR,

    IMAGE_DIR,

    SESSION_DIR,

    REPORT_DIR,

    JSON_DIR,

    LOG_DIR

]


def create_output_directories():
    """
    Create all output directories if they do not exist.
    """

    for folder in DIRECTORIES:
        folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def print_configuration():

    print("=" * 70)
    print(PROJECT_NAME)
    print("Version:", VERSION)
    print("=" * 70)

    print(f"Output folder : {OUTPUT_DIR}")
    print(f"Contact cutoff: {CONTACT_DISTANCE:.1f} Å")
    print(f"HBond cutoff  : {HBOND_MAX_DISTANCE:.1f} Å")
    print(f"Hydrophobic   : {HYDROPHOBIC_DISTANCE:.1f} Å")
    print(f"π-π cutoff    : {PI_STACK_MAX_DISTANCE:.1f} Å")
    print(f"π-cation      : {PI_CATION_MAX_DISTANCE:.1f} Å")
    print("=" * 70)


# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":

    create_output_directories()

    print_configuration()
