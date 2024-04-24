#!/usr/bin/env python3
import argparse
import csv
import logging
import random
import textwrap
from pathlib import Path

from tqdm import tqdm

from phenobase.pylib import log, util

# TODO: Run this in parallel


def main():
    log.started()
    args = parse_args()

    paths = get_image_paths(args.image_dir)

    train_set, val_split, test_split = split_images(
        paths, args.seed, args.train_split, args.val_split, args.test_split
    )

    write_csv(args.split_csv, train_set, val_split, test_split)

    log.finished()


def get_image_paths(image_dir: Path) -> list[Path]:
    paths = list(image_dir.glob("*"))
    valid = [s for s in tqdm(paths) if util.get_sheet_image(s)]

    msg = f"{len(valid)} / {len(paths)} with {len(paths) - len(valid)} errors"
    logging.info(msg)

    return valid


def split_images(
    paths: list[Path],
    seed: int,
    train_split: float,
    val_split: float,
    test_split: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    random.seed(seed)
    random.shuffle(paths)

    total = len(paths)
    split1 = round(total * train_split)
    split2 = split1 + round(total * val_split)

    train_set = paths[:split1]
    val_set = paths[split1:split2]
    test_set = paths[split2:]

    return train_set, val_set, test_set


def write_csv(
    split_csv: Path, train_set: list[Path], val_set: list[Path], test_set: list[Path]
) -> None:
    with split_csv.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split"])

        for path in train_set:
            writer.writerow([str(path), "train"])

        for path in val_set:
            writer.writerow([str(path), "val"])

        for path in test_set:
            writer.writerow([str(path), "test"])


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """
            Split the data into training, testing, & validation datasets.
            """
        ),
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read images for the MAE in this directory.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the MAE splits to this CSV file.""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=2114987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    # Validate splits
    if sum((args.train_split, args.val_split, args.test_split)) != 1.0:
        msg = "train, val, and test splits must sum to 1.0"
        raise ValueError(msg)

    if any(
        s < 0.0 or s > 1.0 for s in (args.train_split, args.val_split, args.test_split)
    ):
        msg = "All splits must be [0.0, 1.0]"
        raise ValueError(msg)

    return args


if __name__ == "__main__":
    main()
