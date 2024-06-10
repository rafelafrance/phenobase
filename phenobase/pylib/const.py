from pathlib import Path

# #########################################################################
CURR_DIR = Path()
IS_SUBDIR = CURR_DIR.name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"

# #########################################################################
TRAITS = " flowering fruiting leaf_out ".split()
TRAIT_2_INT = {t: i for i, t in enumerate(TRAITS)}
TRAIT_2_STR = dict(enumerate(TRAITS))
