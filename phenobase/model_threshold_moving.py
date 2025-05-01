#!/usr/bin/env python

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pylib import log
from sklearn import metrics

STEP = 0.05


@dataclass
class Best:
    accuracy: float
    threshold_lo: float
    threshold_hi: float
    checkpoint: str = ""


def main(args):
    log.started(args=args)

    with args.score_csv.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict] = list(reader)

    checkpoints = [c for c in rows[0] if c.find("checkpoint") > -1]
    trait = rows[0]["trait"]

    all_trues = [float(r[trait]) for r in rows]

    bests = []
    for checkpoint in checkpoints:
        all_scores = [float(r[checkpoint]) for r in rows]
        cp_best = checkpoint_best(checkpoint, all_trues, all_scores, args.pos_limit)
        if cp_best:
            bests.append(cp_best)

    best = max_best(bests)

    print(
        textwrap.dedent(
            f"""
            Trait:          {trait}
            Checkpoint:     {best.checkpoint}
            Accuracy:       {best.accuracy:0.3f}
            Low Threshold:  {best.threshold_lo:0.2f}
            High Threshold: {best.threshold_hi:0.2f}
        """
        )
    )

    all_scores = [float(r[best.checkpoint]) for r in rows]
    y_trues, y_preds = get_preds(all_trues, all_scores)
    all_count = len(y_trues)
    cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
    tn, fp, fn, tp = cells
    print(
        textwrap.dedent(
            f"""
            Starting metrics:
              tp={tp:4d}, fp={fp:4d}
              fn={fn:4d}, tn={tn:4d}
            Starting accuracy  = {metrics.accuracy_score(y_trues, y_preds):0.3f}
            Starting f1        = {metrics.f1_score(y_trues, y_preds):0.3f}
            Starting f0.5      = {metrics.fbeta_score(y_trues, y_preds, beta=0.5):0.3f}
            Starting precision = {metrics.precision_score(y_trues, y_preds):0.3f}
            Starting recall    = {metrics.recall_score(y_trues, y_preds):0.3f}
            Starting count     = {all_count}
        """
        )
    )

    y_trues, y_preds = get_preds(
        all_trues, all_scores, best.threshold_lo, best.threshold_hi
    )
    cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
    tn, fp, fn, tp = cells
    count = len(y_trues)
    print(
        textwrap.dedent(
            f"""
            Best metrics:
              tp={tp:4d}, fp={fp:4d}
              fn={fn:4d}, tn={tn:4d}
            Best accuracy  = {metrics.accuracy_score(y_trues, y_preds):0.3f}
            Best f1        = {metrics.f1_score(y_trues, y_preds):0.3f}
            Best f0.5      = {metrics.fbeta_score(y_trues, y_preds, beta=0.5):0.3f}
            Best precision = {metrics.precision_score(y_trues, y_preds):0.3f}
            Best recall    = {metrics.recall_score(y_trues, y_preds):0.3f}
            Best count     = {count}  fraction of original {(count / all_count):0.3f}
        """
        )
    )

    log.finished()


def checkpoint_best(checkpoint, all_trues, all_scores, pos_limit):
    all_trues, all_preds = get_preds(all_trues, all_scores)
    all_count = len(all_trues)

    all_tn, _fp, _fn, all_tp = metrics.confusion_matrix(all_trues, all_preds).ravel()
    if all_tn == 0 or all_tp == 0:
        return None

    accuracies = []

    for thresh_lo in np.arange(0.0, 0.5 + STEP, STEP):
        thresh_lo = float(thresh_lo)  # squash linters, float != np.floating

        for thresh_hi in np.arange(0.5, 1.0 + STEP, STEP):
            thresh_hi = float(thresh_hi)  # squash linters, float != np.floating

            y_trues, y_preds = get_preds(all_trues, all_scores, thresh_lo, thresh_hi)

            if (len(y_trues) / all_count) < pos_limit:
                continue

            cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
            if len(cells) < 4:
                continue

            tn, fp, fn, tp = cells

            if (tp / all_tp) >= pos_limit and (tn / all_tn) >= pos_limit:
                accuracy = metrics.accuracy_score(y_trues, y_preds)

                new_best = Best(
                    accuracy=accuracy,
                    threshold_lo=thresh_lo,
                    threshold_hi=thresh_hi,
                    checkpoint=checkpoint,
                )
                accuracies.append(new_best)
    if not accuracies:
        return None

    best = max_best(accuracies)
    return best


def get_preds(y_true, y_scores, threshold_lo=0.5, threshold_hi=0.5):
    trues, preds = [], []
    for true, score in zip(y_true, y_scores, strict=True):
        if score < threshold_lo:
            trues.append(true)
            preds.append(0.0)
        elif score >= threshold_hi:
            trues.append(true)
            preds.append(1.0)
    return trues, preds


def max_best(bests):
    return max(bests, key=lambda b: (b.accuracy, b.threshold_hi - b.threshold_lo))


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
        "--pos-limit",
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
