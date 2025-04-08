#!/usr/bin/env python

import argparse
import csv
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pylib import log, util
from sklearn import metrics


@dataclass
class Family:
    family: str
    total: int = 0
    filtered: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    f1: float = np.nan
    f_beta_05: float = np.nan
    recall: float = np.nan
    accuracy: float = np.nan
    precision: float = np.nan


def main(args):
    log.started()

    with args.score_csv.open() as fin:
        reader = csv.DictReader(fin)
        all_rows = [row for row in reader if row["checkpoint"] == args.checkpoint]

    fams = defaultdict(list)
    for row in all_rows:
        fams[row["family"].lower()].append(row)

    # Calculate total metrics
    total = Family("Total")
    total.total = len(all_rows)

    y_trues = [float(r[args.trait]) for r in all_rows]
    y_scores = [float(r[f"{args.trait}_score"]) for r in all_rows]
    y_trues, y_preds = get_preds(
        y_trues, y_scores, args.low_threshold, args.high_threshold
    )
    total.filtered = len(y_preds)

    total.tp = sum(
        int(t == 1 and p == 1) for t, p in zip(y_trues, y_preds, strict=True)
    )
    total.tn = sum(
        int(t == 0 and p == 0) for t, p in zip(y_trues, y_preds, strict=True)
    )
    total.fp = sum(
        int(t == 0 and p == 1) for t, p in zip(y_trues, y_preds, strict=True)
    )
    total.fn = sum(
        int(t == 1 and p == 0) for t, p in zip(y_trues, y_preds, strict=True)
    )

    total.f1 = metrics.f1_score(y_trues, y_preds, zero_division=np.nan)
    total.f_beta_05 = metrics.fbeta_score(
        y_trues, y_preds, beta=0.5, zero_division=np.nan
    )
    total.recall = metrics.recall_score(y_trues, y_preds, zero_division=np.nan)
    total.accuracy = metrics.accuracy_score(y_trues, y_preds)
    total.precision = metrics.precision_score(y_trues, y_preds, zero_division=np.nan)

    # Per family metrics
    families = []
    for name, rows in fams.items():
        fam = Family(name)
        fam.total = len(rows)

        y_trues = [float(r[args.trait]) for r in rows]
        y_scores = [float(r[f"{args.trait}_score"]) for r in rows]
        y_trues, y_preds = get_preds(
            y_trues, y_scores, args.low_threshold, args.high_threshold
        )
        fam.filtered = len(y_preds)

        if fam.filtered == 0:
            continue

        fam.tp = sum(
            int(t == 1 and p == 1) for t, p in zip(y_trues, y_preds, strict=True)
        )
        fam.tn = sum(
            int(t == 0 and p == 0) for t, p in zip(y_trues, y_preds, strict=True)
        )
        fam.fp = sum(
            int(t == 0 and p == 1) for t, p in zip(y_trues, y_preds, strict=True)
        )
        fam.fn = sum(
            int(t == 1 and p == 0) for t, p in zip(y_trues, y_preds, strict=True)
        )

        fam.f1 = metrics.f1_score(y_trues, y_preds, zero_division=np.nan)
        fam.f_beta_05 = metrics.fbeta_score(
            y_trues, y_preds, beta=0.5, zero_division=np.nan
        )
        fam.recall = metrics.recall_score(y_trues, y_preds, zero_division=np.nan)
        fam.accuracy = metrics.accuracy_score(y_trues, y_preds)
        fam.precision = metrics.precision_score(y_trues, y_preds, zero_division=np.nan)

        families.append(fam)

    families = sorted(families, key=lambda f: f.family)

    with args.family_csv.open("w") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "family",
                "total",
                "filtered",
                "tp",
                "fn",
                "fp",
                "tn",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fbeta_0.5",
                "trait",
                "checkpoint",
                "low_threshold",
                "high_threshold",
            ]
        )
        writer.writerow(
            [
                total.family,
                total.total,
                total.filtered,
                total.tp,
                total.fn,
                total.fp,
                total.tn,
                f"{total.accuracy:0.3f}",
                f"{total.precision:0.3f}",
                f"{total.recall:0.3f}",
                f"{total.f1:0.3f}",
                f"{total.f_beta_05:0.3f}",
                args.trait,
                args.checkpoint,
                args.low_threshold,
                args.high_threshold,
            ]
        )

        for fam in families:
            writer.writerow(
                [
                    fam.family,
                    fam.total,
                    fam.filtered,
                    fam.tp,
                    fam.fn,
                    fam.fp,
                    fam.tn,
                    f"{fam.accuracy:0.3f}",
                    f"{fam.precision:0.3f}",
                    f"{fam.recall:0.3f}",
                    f"{fam.f1:0.3f}",
                    f"{fam.f_beta_05:0.3f}",
                ]
            )

    log.finished()


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


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Look at the scores for a checkpoint by family."""
        ),
    )

    arg_parser.add_argument(
        "--score-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This CSV contains all of the test herbarium sheets scored using all of
            the model checkpoints.""",
    )

    arg_parser.add_argument(
        "--family-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the results to this CSV file.""",
    )

    arg_parser.add_argument(
        "--checkpoint",
        required=True,
        help="""The checkpoint to examine.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        required=True,
        help="""The trait to examine.""",
    )

    arg_parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.05,
        help="""The threshold for a negative classification.""",
    )

    arg_parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.95,
        help="""The threshold for a positive classification.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
