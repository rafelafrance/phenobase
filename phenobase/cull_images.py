#!/usr/bin/env python3
import argparse
import logging
import textwrap
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from phenobase.pylib import log, util


def main():
    log.started()
    args = parse_args()

    paths = get_image_paths(args.sheets_dir)
    valid = paths.copy()

    if args.cull_bad_paths:
        valid = cull_bad_paths(valid)

    if args.cull_bad_images:
        valid = cull_bad_images(valid)

    move_culled_images(args.cull_dir, paths, valid)

    log.finished()


def get_image_paths(sheets_dir: Path) -> list[Path]:
    paths = list(sheets_dir.glob("*"))
    return paths


def cull_bad_paths(paths: list[Path]) -> list[Path]:
    """Good paths contain a valid UUID as the file stem."""
    valid = []
    for path in paths:
        try:
            _ = UUID(path.stem)
            valid.append(path)
        except ValueError:
            pass

    log_error_count("Bad paths", valid, paths)

    return valid


# TODO: Run this in parallel
def cull_bad_images(paths: list[Path]) -> list[Path]:
    valid = [s for s in tqdm(paths) if util.get_sheet_image(s)]

    log_error_count("Bad images", valid, paths)

    return valid


def move_culled_images(cull_dir: Path, paths: list[Path], valid: list[Path]) -> None:
    good = {p.stem for p in valid}
    for path in paths:
        if path.stem not in good:
            path.rename(cull_dir / path.name)

    log_error_count("Total culled", valid, paths)


def log_error_count(what: str, valid: list[Path], paths: list[Path]) -> None:
    v, n = len(valid), len(paths)
    msg = f"{what}: {v} / {n} with {n - v} errors"
    logging.info(msg)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent("""Cull bad images from the image directory."""),
    )

    arg_parser.add_argument(
        "--sheets-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read herbarim sheets from this directory.""",
    )

    arg_parser.add_argument(
        "--cull-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Move culled images into this directory.""",
    )

    arg_parser.add_argument(
        "--cull-bad-paths",
        action="store_true",
        help="""Remove paths where the file stem is not a valid GUID.""",
    )

    arg_parser.add_argument(
        "--cull-bad-images",
        action="store_true",
        help="""Remove paths where the image cannot be opened.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
