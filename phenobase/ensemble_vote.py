#!/usr/bin/env python3

import argparse
import csv
import json
import sys
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
    vote_tally = get_vote_tally(args.score_csv, ensemble, pred_col)

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
    df = df[["gbifid", "sci_name", "votes", "winner", "family", "genus", "tiebreaker"]]

    df["family"] = df["family"].str.title()
    df["genus"] = df["genus"].str.title()
    df["winner"] = df["winner"].astype(str)

    mode, header = ("a", False) if args.vote_csv.exists() else ("w", True)
    df.to_csv(args.vote_csv, mode=mode, header=header, index=False)

    print_results(winners)

    log.finished()


def print_results(winners):
    global_vote = Counter(winners.values())
    total = sum(v for v in global_vote.values())
    for value, count in global_vote.items():
        print(f"{value!s:>5} {count:5} {count / total * 100.0:5.2f}")
    print(f"Total {total:5}")


def get_vote_tally(score_csvs, ensemble, pred_col):
    vote_tally = defaultdict(list)

    for model, score_csv in zip(ensemble["checkpoints"], score_csvs, strict=True):
        # Get all votes for a model
        with score_csv.open() as f:
            reader = csv.DictReader(f)
            checkpoint_scores = list(reader)

        duplicates = set()

        for score_row in checkpoint_scores:
            image_path = score_row["path"]
            score = float(score_row[pred_col])

            # Make sure we only get an image's score once per model
            # It happened once so the check is here now
            if image_path in duplicates:
                print(image_path)
                sys.exit()
            duplicates.add(image_path)

            # Convert a score into a vote for the model
            if score < model["threshold_low"]:
                vote = 0
            elif score >= model["threshold_high"]:
                vote = 1
            else:
                vote = None

            vote_tally[image_path].append(vote)

    return vote_tally


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Calculate ensemble results on inferred data.
            These are "hard vote" ensembles, so I'm looking for a majority of votes
            after using model thresholds.
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
        "--vote-csv",
        type=Path,
        metavar="PATH",
        help="""Append vote results to this CSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
