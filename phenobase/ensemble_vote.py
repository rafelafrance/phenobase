#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import logging
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from phenobase.pylib import log


def main(args: argparse.Namespace) -> None:
    log.started(args=args)

    ensemble = get_ensemble(args.ensemble_json)

    votes = gather_votes(args.glob, ensemble)

    tally_votes(votes, ensemble)

    write_csv(args.vote_csv, votes)

    print_results(votes)

    log.finished()


def write_csv(vote_csv: Path, votes) -> None:
    logging.info("Writing CSV file")
    df = pd.DataFrame(votes.values())
    df.to_csv(vote_csv, index=False)


def gather_votes(globs: list[str], ensemble: dict) -> dict[str, dict]:
    votes = {}

    pred_col = ensemble["trait"] + "_score"  # Column that holds the trait's score
    len_ = len(ensemble["checkpoints"])

    for i, glob_ in enumerate(globs):
        checkpoint = ensemble["checkpoints"][i]
        threshold_low = checkpoint["threshold_low"]
        threshold_high = checkpoint["threshold_high"]

        for path in sorted(glob.glob(glob_)):  # noqa: PTH207
            path = Path(path)
            logging.info(path.stem)
            with path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["id"] not in votes:
                        votes[row["id"]] = {
                            "id": row["id"],
                            "trait": ensemble["trait"],
                            "scores": [None for _ in range(len_)],
                            "votes": [None for _ in range(len_)],
                            "winner": None,
                        }
                    score = float(row[pred_col])
                    votes[row["id"]]["scores"][i] = score

                    if score < threshold_low:
                        vote = 0
                    elif score >= threshold_high:
                        vote = 1
                    else:
                        vote = None
                    votes[row["id"]]["votes"][i] = vote

    return votes


def tally_votes(votes: dict[str, dict], ensemble: dict) -> None:
    logging.info("Tally votes")
    vote_threshold = int(ensemble["args"]["vote_threshold"])

    for row in tqdm(votes.values()):
        top = Counter(row["votes"]).most_common()
        best = top[0][0]
        if best is not None and top[0][1] >= vote_threshold:
            row["winner"] = best


def get_ensemble(ensemble_json: Path):
    with ensemble_json.open() as f:
        return json.load(f)


def print_results(votes):
    logging.info("Summarizing")
    global_votes = defaultdict(int)
    for vote in tqdm(votes.values()):
        global_votes[vote["winner"]] += 1
    for value, count in global_votes.items():
        logging.info(f"{value!s:>5} {count:12,d} {count / len(votes) * 100.0:5.2f}%")
    logging.info(f"Total {len(votes):12,d}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Calculate ensemble results on inferred data.
            These are "hard vote" ensembles, so I'm looking for a majority of votes
            after using model thresholds. Note: --glob must be in the same order as the
            models in --ensemble-json. That is, if the best combo has checkpoints:
                "vit_384_lg_flowers_f1_slurm_sl/checkpoint-9450",
                "vit_384_lg_flowers_f1_a/checkpoint-19199",
                "effnet_528_flowers_reg_f1_a/checkpoint-17424"
            then the first --glob must match the outputs of model_infer.py:
                --checkpoint vit_384_lg_flowers_f1_slurm_sl/checkpoint-9450
            It sounds more complicated than it really is. You just need to match the
            glob with the checkpoint models in the ensemble.
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
        "--glob",
        action="append",
        required=True,
        metavar="GLOB",
        help="""A glob for all the output CSVs for a checkpoint in the ensemble.""",
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
