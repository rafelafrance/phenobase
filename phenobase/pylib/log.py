import logging
import sys
from pathlib import Path


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def module_name() -> str:
    return Path(sys.argv[0]).stem


def started() -> None:
    setup_logger()
    logging.info("=" * 80)
    msg = f"{module_name()} started"
    logging.info(msg)


def finished() -> None:
    msg = f"{module_name()} finished"
    logging.info(msg)
