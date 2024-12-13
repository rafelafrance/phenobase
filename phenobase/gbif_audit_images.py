#!/usr/bin/env python3

import argparse
import logging
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pylib import log


@dataclass
class Dupe:
    name: str  # Path.name
    path: Path


Dupes = dict[str, list[Dupe]]


def main():
    log.started()

    args = parse_args()
    logging.info(args)

    all_dupes: Dupes = get_paths(args.src_dir)

    report(all_dupes)
    choose_file(all_dupes)
    remove_duplicates(all_dupes)

    log.finished()


def get_paths(src_dir: Path) -> Dupes:
    all_dupes: Dupes = defaultdict(list)

    for dir_ in src_dir.glob("images_*"):
        for path in dir_.glob("*.jpg"):
            gbifid, tiebreaker, *_ = path.stem.split("_", maxsplit=2)
            key = f"{gbifid}_{tiebreaker}"
            all_dupes[key].append(Dupe(path.name, path))

    return all_dupes


def choose_file(all_dupes: Dupes) -> None:
    """
    Get the best file when there are multiple attempts to download a file.

    A happy coincidence is that the shortest file name is the best. Name formats:
        - <gbif ID>.jpg
        - <gbif ID>_small.jpg
        - <gbif ID>_image_error.jpg   * This one is just as bad as the next one *
        - <gbif ID>_download_error.jpg
    """
    for key, dupes in all_dupes.items():
        if len(dupes) > 1 and any(dupe.name != dupes[0].name for dupe in dupes):
            all_dupes[key] = sorted(dupes, key=lambda dupe: len(dupe.name))


def remove_duplicates(all_dupes: Dupes) -> None:
    for dupes in all_dupes.values():
        # Everything after the first image gets deleted
        for dupe in dupes[1:]:
            dupe.path.unlink(missing_ok=True)


def report(all_dupes: Dupes) -> None:
    total = len(all_dupes)
    logging.info(f"{'Total':<15} {total:8,d}")

    dupe_count = sum(1 for dupes in all_dupes.values() if len(dupes) > 1)
    logging.info(f"{'Duplicates':<15} {dupe_count:8,d}")

    differ = sum(
        1
        for dupes in all_dupes.values()
        if len(dupes) > 1 and any(dupe.name != dupes[0].name for dupe in dupes)
    )
    logging.info(f"{'Different':<15} {differ:8,d}")

    small = sum(
        1
        for dupes in all_dupes.values()
        if any(dupe.name.find("small") > -1 for dupe in dupes)
    )
    logging.info(f"{'Small images':<15} {small:8,d}")

    down = sum(
        1
        for dupes in all_dupes.values()
        if any(dupe.name.find("download_error") > -1 for dupe in dupes)
    )
    logging.info(f"{'Download errors':<15} {down:8,d}")

    image = sum(
        1
        for dupes in all_dupes.values()
        if any(dupe.name.find("image_error") > -1 for dupe in dupes)
    )
    logging.info(f"{'Image errors':<15} {image:8,d}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Audit the downloaded images."""),
    )

    arg_parser.add_argument(
        "--src-dir",
        type=Path,
        metavar="PATH",
        help="""Source image directory. It contains subdirectories with images.""",
    )

    arg_parser.add_argument(
        "--dst-dir",
        type=Path,
        metavar="PATH",
        help="""Destination image directory.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
