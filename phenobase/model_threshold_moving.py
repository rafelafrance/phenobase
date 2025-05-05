#!/usr/bin/env python

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from pylib import log, util
from sklearn import metrics
from tqdm import tqdm

STEP = 0.05


def main(args):
    log.started(args=args)

    df1 = pd.read_csv(args.score_csv)

    checkpoints = [c for c in df1.columns if c.find("checkpoint") > -1]
    trait = df1.loc[0, "trait"]

    all_trues = df1.loc[:, trait]

    df1 = df1.set_index("name")

    df2 = pd.DataFrame(columns=checkpoints, index=util.INDEXES)

    bests = []
    for checkpoint in tqdm(checkpoints):
        all_scores = df1.loc[:, checkpoint]
        best = best_thresholds(all_trues, all_scores, args.remove_limit)

        for index, value in best.items():
            df2.loc[index, checkpoint] = value

        best["checkpoint"] = checkpoint
        bests.append(best)

    bests = sort_bests(bests)
    for i, best in enumerate(bests, 1):
        print(f"{i} {best.get('accuracy', 0.0):5.3f} {best['checkpoint']}")

    df2.to_csv(args.output_csv, index_label="metric")

    log.finished()


def filter_scores(y_trues, y_scores, threshold_lo=0.5, threshold_hi=0.5):
    trues, preds = [], []
    for true, score in zip(y_trues, y_scores, strict=True):
        if score < threshold_lo:
            trues.append(true)
            preds.append(0.0)
        elif score >= threshold_hi:
            trues.append(true)
            preds.append(1.0)
    return trues, preds


def best_thresholds(all_trues, all_scores, remove_limit) -> dict:
    all_trues, all_preds = filter_scores(all_trues, all_scores)
    all_count = len(all_trues)

    all_tn, _fp, _fn, all_tp = metrics.confusion_matrix(all_trues, all_preds).ravel()
    if all_tn == 0 or all_tp == 0:
        return {"error": "Missing tp or tn"}

    bests = []

    for thresh_lo in np.arange(0.0, 0.5 + STEP, STEP):
        thresh_lo = float(thresh_lo)  # squash linters, float != np.floating

        for thresh_hi in np.arange(0.5, 1.0 + STEP, STEP):
            thresh_hi = float(thresh_hi)  # squash linters, float != np.floating

            y_trues, y_preds = filter_scores(
                all_trues, all_scores, thresh_lo, thresh_hi
            )

            if (len(y_trues) / all_count) < remove_limit:
                continue

            cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
            if len(cells) < 4:
                continue

            tn, fp, fn, tp = cells

            if (tp / all_tp) >= remove_limit and (tn / all_tn) >= remove_limit:
                bests.append(
                    {
                        "threshold_low": thresh_lo,
                        "threshold_high": thresh_hi,
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "count": len(y_trues),
                        "fraction": len(y_trues) / all_count,
                        "accuracy": metrics.accuracy_score(y_trues, y_preds),
                        "f1": metrics.f1_score(y_trues, y_preds),
                        "f0.5": metrics.fbeta_score(y_trues, y_preds, beta=0.5),
                        "precision": metrics.precision_score(y_trues, y_preds),
                        "recall": metrics.recall_score(y_trues, y_preds),
                    }
                )

    if not bests:
        return {"error": "Not enough tp or tn in the thresholds"}

    bests = sort_bests(bests)
    return bests[0]


def sort_bests(bests):
    return sorted(
        bests,
        reverse=True,
        key=lambda b: (
            b.get("accuracy", 0.0),
            b.get("threshold_high", 0.0) - b.get("threshold_low", 0.0),
        ),
    )


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Move thresholds to find the best models and thresholds for each trait.

            Find high and low score thresholds for each trait model that maximize its
            scores. That is we will throw out records that are outside of the
            thresholds because the model "wasn't really sure" about its classifiations.
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
        "--output-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Out put the score results to this CSV.""",
    )

    arg_parser.add_argument(
        "--remove-limit",
        type=float,
        metavar="FRACTION",
        default=0.7,
        help="""Do not remove more than this fraction of true positives &
            true negatives. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
