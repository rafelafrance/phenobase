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

    paths = list(args.sheets_dir.glob("*"))
    good_images = paths.copy()

    if args.resize_dir:
        args.resize_dir.mkdir(parents=True, exist_ok=True)

    if args.cull_bad_paths:
        good_images = cull_bad_paths(good_images)

    if args.cull_bad_images:
        good_images = cull_bad_images(good_images, args.resize_dir, args.resize)

    move_culled_images(args.cull_dir, paths, good_images)

    log.finished()


def cull_bad_paths(paths: list[Path]) -> list[Path]:
    """Good paths contain a good_images UUID as the file stem."""
    good_images = []
    for path in paths:
        try:
            _ = UUID(path.stem)
            good_images.append(path)
        except ValueError:  # noqa: PERF203
            continue

    log_error_count("Bad paths", good_images, paths)
    return good_images


# TODO: Run this in parallel
def cull_bad_images(paths: list[Path], resize_dir: Path, resize: int) -> list[Path]:
    good_images = []
    for path in tqdm(paths):
        image = util.get_sheet_image(path)
        if image:
            good_images.append(path)
            if resize_dir:
                image = util.resize_image(image, resize)
                image.save(resize_dir / path.name)

    log_error_count("Bad images", good_images, paths)
    return good_images


def move_culled_images(
    cull_dir: Path, paths: list[Path], good_images: list[Path]
) -> None:
    good = {p.stem for p in good_images}
    for path in paths:
        if path.stem not in good:
            path.rename(cull_dir / path.name)

    log_error_count("Total culled", good_images, paths)


def log_error_count(what: str, good_images: list[Path], paths: list[Path]) -> None:
    v, n = len(good_images), len(paths)
    msg = f"{what}: {v} / {n} with {n - v} errors"
    logging.info(msg)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Cull bad images from the image directory."""),
    )

    arg_parser.add_argument(
        "--sheets-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read herbarium sheets from this directory.""",
    )

    arg_parser.add_argument(
        "--cull-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Move culled images into this directory.""",
    )

    arg_parser.add_argument(
        "--resize-dir",
        metavar="PATH",
        type=Path,
        help="""Write resized images to this directory. Leave it blank if you do not
            want to resize images.""",
    )

    arg_parser.add_argument(
        "--resize",
        type=int,
        metavar="INT",
        default=224,
        help="""Resize images to this size (pixels). (default: %(default)s)""",
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
