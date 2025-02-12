#!/usr/bin/env python

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pylib import const, log
from sklearn import metrics


@dataclass
class Best:
    accuracy: float
    threshold_lo: float
    threshold_hi: float
    trait: str
    checkpoint: str = ""


def main(args):
    log.started()

    df_all = pd.read_csv(args.score_csv, low_memory=False)
    df_all = filter_families(df_all, args.bad_families)

    checkpoints = df_all["checkpoint"].unique()

    bests = []
    for checkpoint in checkpoints:
        df = df_all.loc[(df_all["checkpoint"] == checkpoint)]
        if len(df) == 0:
            continue

        all_trues, all_scores = get_raw_data(df, args.trait)
        if all_trues is None or len(all_trues) == 0 or all_scores.hasnans:
            continue

        cp_best = checkpoint_best(all_trues, all_scores, args.trait, args.pos_limit)
        cp_best.checkpoint = checkpoint
        bests.append(cp_best)

    best = max_best(bests)

    print(
        f"""
        Trait:          {best.trait}
        Checkpoint:     {best.checkpoint}
        Accuracy:       {best.accuracy:0.3f}
        Low Threshold:  {best.threshold_lo:0.2f}
        High Threshold: {best.threshold_hi:0.2f}
    """
    )

    df = df_all.loc[df_all["checkpoint"] == best.checkpoint]
    y_trues, y_scores = get_raw_data(df, args.trait)

    y_trues, y_preds = get_preds(y_trues, y_scores)
    all_count = len(y_trues)
    print("Starting metrics")
    print(metrics.confusion_matrix(y_trues, y_preds))
    print(f"Starting accuracy  = {metrics.accuracy_score(y_trues, y_preds):0.3f}")
    print(f"Starting f1        = {metrics.f1_score(y_trues, y_preds):0.3f}")
    beta = metrics.fbeta_score(y_trues, y_preds, beta=0.5)
    print(f"Starting f0.5      = {beta:0.3f}")
    print(f"Starting precision = {metrics.precision_score(y_trues, y_preds):0.3f}")
    print(f"Starting recall    = {metrics.recall_score(y_trues, y_preds):0.3f}")
    print(f"Starting count     = {all_count}")
    print()

    y_trues, y_preds = get_preds(
        y_trues, y_scores, best.threshold_lo, best.threshold_hi
    )
    count = len(y_trues)
    print("Best metrics")
    print(metrics.confusion_matrix(y_trues, y_preds))
    print(f"Best accuracy  = {metrics.accuracy_score(y_trues, y_preds):0.3f}")
    print(f"Best f1        = {metrics.f1_score(y_trues, y_preds):0.3f}")
    beta = metrics.fbeta_score(y_trues, y_preds, beta=0.5)
    print(f"Best f0.5      = {beta:0.3f}")
    print(f"Best precision = {metrics.precision_score(y_trues, y_preds):0.3f}")
    print(f"Best recall    = {metrics.recall_score(y_trues, y_preds):0.3f}")
    print(f"Best count     = {count}  fraction of orginal = {(count / all_count):0.3f}")
    print()

    log.finished()


def max_best(bests):
    return max(bests, key=lambda b: (b.accuracy, b.threshold_hi - b.threshold_lo))


def filter_families(df, bad_families):
    if not bad_families:
        return df
    bad = pd.read_csv(bad_families)
    df = df.loc[~df["family"].isin(bad["family"]), :]
    return df


def get_raw_data(df, trait):
    try:
        y_trues = df[trait].astype(float)
        y_scores = df[f"{trait}_score"].astype(float)
    except ValueError:
        return None, None
    return y_trues, y_scores


def checkpoint_best(all_trues, all_scores, trait, pos_limit):
    all_trues, all_preds = get_preds(all_trues, all_scores)

    all_tn, all_fp, all_fn, all_tp = metrics.confusion_matrix(
        all_trues, all_preds
    ).ravel()

    all_count = len(all_trues)

    accuracies = []
    for thresh_lo in np.arange(0.0, 0.5, 0.05):
        for thresh_hi in np.arange(0.5, 1.0, 0.05):
            y_trues, y_preds = get_preds(all_trues, all_scores, thresh_lo, thresh_hi)

            if (len(y_trues) / all_count) < pos_limit:
                continue

            cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
            if len(cells) < 4:
                continue

            tn, fp, fn, tp = cells

            if (tp / all_tp) > pos_limit and (tn / all_tn) > pos_limit:
                accuracy = metrics.accuracy_score(y_trues, y_preds)

                accuracies.append(
                    Best(
                        accuracy=accuracy,
                        threshold_lo=thresh_lo,
                        threshold_hi=thresh_hi,
                        trait=trait,
                    )
                )
    best = max_best(accuracies)
    return best


def get_preds(y_trues, y_scores, threshold_lo=0.5, threshold_hi=0.5):
    trues, preds = [], []
    for true, score in zip(y_trues, y_scores, strict=True):
        if score < threshold_lo:
            trues.append(true)
            preds.append(0.0)
        elif score >= threshold_hi:
            trues.append(true)
            preds.append(1.0)
    return trues, preds


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
        "--trait",
        choices=const.TRAITS,
        help="""The trait to examine.""",
    )

    arg_parser.add_argument(
        "--pos-limit",
        type=float,
        metavar="FRACTION",
        default=0.7,
        help="""Do not remove more than this fraction of true positives &
            true negatives. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--bad-families",
        metavar="PATH",
        type=Path,
        help="""Make sure records in these families are skipped.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
