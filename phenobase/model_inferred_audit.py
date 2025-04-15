#!/usr/bin/env python3

import argparse
import csv
import shutil
import textwrap
from pathlib import Path
from random import seed, shuffle

import pandas as pd
from tqdm import tqdm

from phenobase.pylib import log, util

DST = Path("/home/rafe.lafrance/blue/phenobase/data/audit_inferred")

TAKE = {
    "neg_equivocal": 200,
    "neg_unequivocal": 200,
    "pos_equivocal": 200,
    "pos_unequivocal": 500,
}


def main(args):
    log.started(args=args)

    seed(args.seed)

    groups = {key: [] for key in TAKE}

    pred_col = f"{args.trait}_score"

    with args.dataset_csv.open() as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="reading"):
            if row.get(pred_col) is None:
                continue
            if Path(row["path"]).stem.endswith("_small"):
                continue

            pred = float(row[pred_col])
            if pred <= args.thresh_low:
                groups["neg_unequivocal"].append(row)
            elif pred < 0.5:
                groups["neg_equivocal"].append(row)
            elif pred >= args.thresh_high:
                groups["pos_unequivocal"].append(row)
            else:
                groups["pos_equivocal"].append(row)

    for key, rows in groups.items():
        dir_ = DST / key
        dir_.mkdir(parents=True, exist_ok=True)

        shuffle(rows)
        rows = rows[: TAKE[key]]
        rows = sorted(rows, key=lambda r: r["gbifid"])

        csv_path = DST / f"{key}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        for row in tqdm(rows, desc=key):
            src = Path(row["path"])
            dst = dir_ / src.name
            if not dst.exists():
                shutil.copy(src, dst)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Get inferred data for auditing results."""),
    )

    arg_parser.add_argument(
        "--dataset-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A CSV file with images and trait labels.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        required=True,
        help="""Infer this trait.""",
    )

    arg_parser.add_argument(
        "--thresh-low",
        type=float,
        default=0.05,
        help="""Threshold for being in the negative class (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--thresh-high",
        type=float,
        default=0.95,
        help="""Threshold for being in the positive class (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=3258221,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
