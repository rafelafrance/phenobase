#!/usr/bin/env python

import argparse
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pylib import const, log
from pylib.binary_metrics import Metrics


@dataclass
class Score:
    score: float
    threshold: float
    checkpoint: str = ""
    trait: str = ""


def main():
    log.started()

    args = parse_args()

    df = pd.read_csv(args.score_csv)

    checkpoints = df["checkpoint"].unique()
    traits = [args.trait] if args.trait else df["trait"].unique()

    scores = score_models(
        df,
        traits,
        checkpoints,
        recall_limit=args.recall_limit,
        min_threshold=args.min_threshold,
    )
    display_best(df, scores, traits, count=args.count)

    log.finished()


def score_models(df, traits, checkpoints, recall_limit, min_threshold):
    scores = defaultdict(list)
    for trait in traits:
        for checkpoint in checkpoints:
            df1 = df.loc[(df["checkpoint"] == checkpoint) & (df["trait"] == trait)]
            if len(df) > 0:
                metrics = Metrics(y_true=df1["y_true"], y_pred=df1["y_pred"])
                score = find_best_score(metrics, recall_limit, min_threshold)
                scores[trait].append(
                    Score(
                        score=score.score,
                        threshold=score.threshold,
                        checkpoint=checkpoint,
                        trait=trait,
                    )
                )
    return {
        k: sorted(v, key=lambda b: b.score, reverse=True) for k, v in scores.items()
    }


def find_best_score(metrics, recall_limit, min_threshold):
    best = Score(0.0, 0.0)
    for threshold in np.arange(min_threshold, 1.0, 0.01):
        metrics.filter_y(thresh_lo=threshold, thresh_hi=threshold)
        if (
            metrics.precision >= best.score
            and metrics.recall >= recall_limit
            and metrics.total > 0
        ):
            best = Score(score=metrics.precision, threshold=threshold)
    return best


def display_best(df, scores, traits, count=5):
    for trait in traits:
        print("-" * 80)
        for i in range(count):
            score = scores[trait][i]
            df1 = df.loc[
                (df["checkpoint"] == score.checkpoint) & (df["trait"] == score.trait)
            ]
            metrics = Metrics(y_true=df1["y_true"], y_pred=df1["y_pred"])
            metrics.filter_y(thresh_lo=score.threshold, thresh_hi=score.threshold)
            print(score)
            print(f"{trait} precision={metrics.precision} recall={metrics.recall}\n")
            metrics.display_matrix()
            print()
        print()


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Move thresholds to find the best models for each trait.

            For a single_label muti-class problem, I'm looking at the value of the
            winning class and remove values that are too close to 0.5. What is the
            cutoff value that maximizes the accuracy scores (or another metric) for the
            model.

            For a regression problem, I'll do roughly the same as above, except that
            there is only one value for each classification.
            """
        ),
    )

    arg_parser.add_argument(
        "--score-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""This CSV contains all of the test herbarium sheets scored using all of
            the model checkpoints.""",
    )

    arg_parser.add_argument(
        "--recall-limit",
        type=float,
        metavar="FRACTION",
        default=0.7,
        help="""The model must have a recall value that is >= to this.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--count",
        type=int,
        metavar="N",
        default=5,
        help="""How many of the top models to display per each trait.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.5,
        help="""The minimum threshold for finding the best model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=const.TRAITS,
        help="""The trait to examine.""",
    )

    arg_parser.add_argument(
        "--problem-type",
        choices=list(const.PROBLEM_TYPES.keys()),
        default="regression",
        help="""This chooses the appropriate scoring method for the type of problem.
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
