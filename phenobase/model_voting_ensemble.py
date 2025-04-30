#!/usr/bin/env python3

import argparse
import csv
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path

from tqdm import tqdm


def merge_scores(args):
    with args.score_csv.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict] = list(reader)
        if args.limit:
            rows = rows[: args.limit]

    score_cols = [c for c in rows[0] if c.find("checkpoint") > -1]
    trait = rows[0]["trait"]

    bests = []

    for combo in tqdm(combinations(score_cols, args.r)):
        correct = 0
        for row in rows:
            true = float(row[trait])
            scores = [round(float(row[c])) for c in combo]
            votes = Counter(scores).most_common(1)
            votes = float(votes[0][0])  # Get the value of the most common vote
            eq = true == votes
            correct += int(eq)
        bests.append((correct, combo))

        if args.limit:
            break

    bests = sorted(bests, reverse=True)
    print(bests[0])
    print(len(rows))


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Vote."""),
    )

    arg_parser.add_argument(
        "--score-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""File that contains all of the model checkpoint scores.""",
    )

    arg_parser.add_argument(
        "--r",
        type=int,
        metavar="INT",
        default=3,
        help="""Voting size is this many checkpoints. (default: %(default)s)""",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit dataset size for testing.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    merge_scores(ARGS)
