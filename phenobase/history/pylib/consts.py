"""Literals used in the system."""
from pathlib import Path

# #########################################################################
CURR_DIR = Path.cwd()
IS_SUBDIR = CURR_DIR.name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"

# #########################################################################
