#!/usr/bin/env python3

import argparse
import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from phenobase.pylib import log


def main(args):
    log.started(args=args)

    # Get ensemble data
    with args.ensemble_json.open() as f:
        ensemble = json.load(f)

    # Tally votes
    pred_col = ensemble["trait"] + "_score"  # Column that holds the trait's score
    vote_tally = get_vote_tally(args, ensemble, pred_col)

    # Get metadata
    with args.score_csv[0].open() as f:
        reader = csv.DictReader(f)
        metadata = {r["path"]: r for r in reader}

    # Get the vote winners for each herbarium sheet
    vote_threshold = int(ensemble["args"]["vote_threshold"])
    winners = {}
    output = []

    for path, votes in vote_tally.items():
        top = Counter(votes).most_common()
        best = top[0][0]
        winner = None
        if best is not None and top[0][1] >= vote_threshold:
            winner = best
        winners[path] = winner
        output.append({"path": path, "votes": votes, "winner": winner} | metadata[path])

    # Output votes and metadata to a CSV file
    df = pd.DataFrame(output)
    df = df.drop([ensemble["trait"], pred_col], axis=1)
    df.to_csv(args.output_csv, index=False)

    print_results(winners)

    log.finished()


def print_results(winners):
    global_vote = Counter(winners.values())
    total = sum(v for v in global_vote.values())
    for value, count in global_vote.items():
        print(f"{value!s:>5} {count:5} {count / total * 100.0:5.2f}")
    print(f"Total {total:5}")


def get_vote_tally(args, ensemble, pred_col):
    vote_tally = defaultdict(list)
    for cp, score_csv in zip(ensemble["checkpoints"], args.score_csv, strict=True):
        low = cp["threshold_low"]
        high = cp["threshold_high"]

        # Get all votes for the checkpoint
        with score_csv.open() as f:
            reader = csv.DictReader(f)
            scores = list(reader)

        duplicates = set()

        for row in scores:
            key = row["path"]
            pred = float(row[pred_col])

            if key in duplicates:
                continue
            duplicates.add(key)

            if pred < low:
                vote = 0.0
            elif pred >= high:
                vote = 1.0
            else:
                vote = None

            vote_tally[key].append(vote)
    return vote_tally


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Calculate ensemble results on inferred data.
            These are "hard vote" ensembles, so I'm looking for a majority of votes
            after thresholds. An expert will check the results.
            """
        ),
    )

    arg_parser.add_argument(
        "--ensemble-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""The JSON file that contains ensemble metrics.""",
    )

    arg_parser.add_argument(
        "--score-csv",
        type=Path,
        required=True,
        action="append",
        metavar="PATH",
        help="""The score CSV files from running the ensemble models. Enter them in the
            same order as they are in the ensemble JSON file.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Output the results to this CSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
