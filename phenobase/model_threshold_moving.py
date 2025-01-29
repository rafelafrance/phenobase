#!/usr/bin/env python

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pylib import const, log


@dataclass
class Best:
    accuracy: float
    threshold_lo: float
    threshold_hi: float
    trait: str
    problem_type: str = ""
    checkpoint: str = ""


def main(args):
    log.started()

    df_all = pd.read_csv(args.score_csv)

    checkpoints = df_all["checkpoint"].unique()

    bests = []
    for checkpoint in checkpoints:
        df = df_all.loc[
            (df_all["checkpoint"] == checkpoint)
            & (df_all["problem_type"] == args.problem_type)
        ]
        if len(df) == 0:
            continue
        if args.problem_type == const.SINGLE_LABEL:
            best = checkpoint_best_single_label(df, args.trait)
            if not best:
                continue
            best.problem_type = args.problem_type
            best.checkpoint = checkpoint
            bests.append(best)
    overall = max_best(bests)
    print(overall)

    # scores = score_models(
    #     df_all,
    #     traits,
    #     checkpoints,
    #     recall_limit=args.recall_limit,
    #     min_threshold=args.min_threshold,
    # )
    # display_best(df_all, scores, traits, count=args.count)

    log.finished()


def max_best(bests):
    return max(bests, key=lambda b: (b.accuracy, b.threshold_hi - b.threshold_lo))


def checkpoint_best_single_label(df, trait: str):
    try:
        scores = df[f"{trait}_score"].astype(float)
        y_trues = df[trait].astype(float)
    except ValueError:
        return None

    accuracies = []
    for threshold_lo in np.arange(0.0, 0.5, 0.1):
        for threshold_hi in np.arange(0.5, 1.0, 0.1):
            y = np.stack((y_trues, scores), axis=1)
            y = y[(y[:, 1] <= threshold_lo) | (y[:, 1] >= threshold_hi)]
            tp = np.where((y[:, 0] == 1.0) & (y[:, 1] >= threshold_hi), 1.0, 0.0).sum()
            fn = np.where((y[:, 0] == 1.0) & (y[:, 1] <= threshold_lo), 1.0, 0.0).sum()
            fp = np.where((y[:, 0] == 0.0) & (y[:, 1] >= threshold_hi), 1.0, 0.0).sum()
            tn = np.where((y[:, 0] == 0.0) & (y[:, 1] <= threshold_lo), 1.0, 0.0).sum()
            acc = (tp + tn) / (tp + tn + fp + fn)
            accuracies.append(
                Best(
                    accuracy=acc,
                    threshold_lo=threshold_lo,
                    threshold_hi=threshold_hi,
                    trait=trait,
                )
            )
    return max_best(accuracies)


# def score_models(df, traits, checkpoints, recall_limit, min_threshold):
#     scores = defaultdict(list)
#     for trait in traits:
#         for checkpoint in checkpoints:
#             df1 = df.loc[(df["checkpoint"] == checkpoint) & (df["trait"] == trait)]
#             if len(df) > 0:
#                 metrics = Metrics(y_true=df1["y_true"], y_pred=df1["y_pred"])
#                 score = find_best_score(metrics, recall_limit, min_threshold)
#                 scores[trait].append(
#                     Score(
#                         score=score.score,
#                         threshold=score.threshold,
#                         checkpoint=checkpoint,
#                         trait=trait,
#                     )
#                 )
#     return {
#         k: sorted(v, key=lambda b: b.score, reverse=True) for k, v in scores.items()
#     }
#
#
# def find_best_score(metrics, recall_limit, min_threshold):
#     best = Score(0.0, 0.0)
#     for threshold in np.arange(min_threshold, 1.0, 0.01):
#         metrics.filter_y(thresh_lo=threshold, thresh_hi=threshold)
#         if (
#             metrics.precision >= best.score
#             and metrics.recall >= recall_limit
#             and metrics.total > 0
#         ):
#             best = Score(score=metrics.precision, threshold=threshold)
#     return best
#
#
# def display_best(df, scores, traits, count=5):
#     for trait in traits:
#         print("-" * 80)
#         for i in range(count):
#             score = scores[trait][i]
#             df1 = df.loc[
#                 (df["checkpoint"] == score.checkpoint) & (df["trait"] == score.trait)
#             ]
#             metrics = Metrics(y_true=df1["y_true"], y_pred=df1["y_pred"])
#             metrics.filter_y(thresh_lo=score.threshold, thresh_hi=score.threshold)
#             print(score)
#             print(f"{trait} precision={metrics.precision} recall={metrics.recall}\n")
#             metrics.display_matrix()
#             print()
#         print()
#


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Move thresholds to find the best modes and thresholds for each trait.

            Find high and low score thresholds for each trait model that maximize its
            scores. That is we will throw out records that are outside of the
            thresholds because the model wasn't really "sure" about its classifiations.
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
        help="""The model must have a recall value that is >= to this. I found that
            the could generate cutoffs that excluded all of one class of values.
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
    ARGS = parse_args()
    main(ARGS)
