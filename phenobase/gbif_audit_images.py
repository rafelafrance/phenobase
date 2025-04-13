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


def main(args):
    log.started(args=args)

    all_dupes: Dupes = get_paths(args.image_dir)
    choose_file(all_dupes)
    removed = remove_duplicate_images(
        all_dupes, remove_duplicates=args.remove_duplicates
    )
    report(all_dupes, removed)

    log.finished()


def get_paths(image_dir: Path) -> Dupes:
    all_dupes: Dupes = defaultdict(list)

    for dir_ in image_dir.glob("images_*"):
        for path in dir_.glob("*.jpg"):
            gbifid, tiebreaker, *_ = path.stem.split("_", maxsplit=2)
            key = f"{gbifid}_{tiebreaker}"
            all_dupes[key].append(Dupe(path.name, path))

    return all_dupes


def choose_file(all_dupes: Dupes) -> None:
    """
    Get the best file when there are multiple attempts to download a file.

    A happy coincidence is that the shortest file name is the best. Name formats:
    * GBIF IDs are all the same length.
        - <gbif ID>.jpg
        - <gbif ID>_small.jpg
        - <gbif ID>_image_error.jpg   * This one is just as bad as the next one *
        - <gbif ID>_download_error.jpg
    """
    for key, dupes in all_dupes.items():
        if len(dupes) > 1 and any(dupe.name != dupes[0].name for dupe in dupes):
            all_dupes[key] = sorted(dupes, key=lambda dupe: len(dupe.name))


def remove_duplicate_images(all_dupes: Dupes, *, remove_duplicates: bool) -> int:
    removed = 0
    for dupes in all_dupes.values():
        # Everything after the first image gets deleted
        for dupe in dupes[1:]:
            removed += 1
            if remove_duplicates:
                dupe.path.unlink(missing_ok=True)
    return removed


def report(all_dupes: Dupes, removed: int) -> None:
    logging.info(f"{'Total':<15} {len(all_dupes):8,d}")

    logging.info(f"{'Removed images':<15} {removed:8,d}")

    small = sum(1 for dupes in all_dupes.values() if dupes[0].name.find("small") > 1)
    logging.info(f"{'Small images':<15} {small:8,d}")

    down = sum(1 for dupes in all_dupes.values() if dupes[0].name.find("download") > 1)
    logging.info(f"{'Download errors':<15} {down:8,d}")

    image = sum(1 for dupes in all_dupes.values() if dupes[0].name.find("image") > 1)
    logging.info(f"{'Image errors':<15} {image:8,d}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Audit the downloaded images."""),
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Directory containing subdirectories with images.""",
    )

    arg_parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="""Destination image directory.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
