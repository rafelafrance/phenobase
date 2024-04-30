#!/usr/bin/env python3
import argparse
import csv
import random
import textwrap
from pathlib import Path

from tqdm import tqdm

from phenobase.pylib import log, util


def main():
    log.started()
    args = parse_args()

    paths = get_image_paths(args.input_dir)

    train_set, val_set, test_set = split_images(
        paths, args.seed, args.train_split, args.val_split, args.test_split
    )

    write_csv(
        args.split_csv, args.output_dir, train_set, val_set, test_set, args.resize
    )

    log.finished()


def get_image_paths(input_dir: Path) -> list[Path]:
    paths = list(input_dir.glob("*.jpg"))
    return paths


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
    split_csv: Path,
    output_dir: Path,
    train_set: list[Path],
    val_set: list[Path],
    test_set: list[Path],
    resize: int,
) -> None:
    with split_csv.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "split"])

        split_dir = output_dir / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in tqdm(train_set, desc="train"):
            writer.writerow([path.name, "train"])
            image = util.get_sheet_image(path)
            image = util.resize_image(image, resize)
            image.save(split_dir / path.name)

        split_dir = output_dir / "val"
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in tqdm(val_set, desc="val  "):
            writer.writerow([path.name, "val"])
            image = util.get_sheet_image(path)
            image = util.resize_image(image, resize)
            image.save(split_dir / path.name)

        split_dir = output_dir / "test"
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in tqdm(test_set, desc="test "):
            writer.writerow([path.name, "test"])
            image = util.get_sheet_image(path)
            image = util.resize_image(image, resize)
            image.save(split_dir / path.name)


def validate_splits(args: argparse.Namespace) -> None:
    if sum((args.train_split, args.val_split, args.test_split)) != 1.0:
        msg = "train, val, and test splits must sum to 1.0"
        raise ValueError(msg)

    if any(
        s < 0.0 or s > 1.0 for s in (args.train_split, args.val_split, args.test_split)
    ):
        msg = "All splits must be [0.0, 1.0]"
        raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """Split the data into training, testing, & validation datasets."""
        ),
    )

    arg_parser.add_argument(
        "--input-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read images for the MAE in this directory.""",
    )

    arg_parser.add_argument(
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Write images for the MAE into this directory.""",
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
        "--resize",
        type=int,
        metavar="INT",
        default=224,
        help="""Resize images for classification. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=2114987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    validate_splits(args)

    return args


if __name__ == "__main__":
    main()
