import logging
import sys
from pathlib import Path


def setup_logger(file_name=None) -> None:
    logging.basicConfig(
        filename=file_name,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def module_name() -> str:
    return Path(sys.argv[0]).stem


def started(file_name=None, *, args=None) -> None:
    setup_logger(file_name)
    logging.info("=" * 80)
    msg = f"{module_name()} started"
    logging.info(msg)
    if args:
        log_args(args)


def finished() -> None:
    msg = f"{module_name()} finished"
    logging.info(msg)


def log_args(args) -> None:
    for key, val in sorted(vars(args).items()):
        logging.info(f"Argument: {key} = {val}")
