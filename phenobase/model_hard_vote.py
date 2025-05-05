#!/usr/bin/env python3

import argparse
import csv
import textwrap
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from pprint import pp

from pylib import log, util
from sklearn import metrics
from tqdm import tqdm


def main(args):
    log.started(args=args)

    with args.metric_csv.open() as f:
        reader = csv.DictReader(f)
        metric: dict = {r["metric"]: r for r in reader}

    with args.score_csv.open() as f:
        reader = csv.DictReader(f)
        scores = list(reader)

    # Trying to keep the number of combos under control
    m_row = metric[args.checkpoint_metric]
    checkpoints = [c for c in m_row if c.find("checkpoint") > -1]
    checkpoints = [
        c for c in checkpoints if m_row[c] and float(m_row[c]) >= args.checkpoint_limit
    ]

    trait = scores[0]["trait"]
    trues = {s["name"]: float(s[trait]) for s in scores}

    preds = defaultdict(dict)
    for row in scores:
        name = row["name"]
        for cp in checkpoints:
            thresh_lo = metric["threshold_low"][cp]
            thresh_hi = metric["threshold_high"][cp]
            score = row[cp]
            if score < thresh_lo:
                preds[cp][name] = "0"
            elif score >= thresh_hi:
                preds[cp][name] = "1"
            else:
                preds[cp][name] = None

    bests = []

    combos = list(combinations(checkpoints, args.votes))

    for combo in tqdm(combos):
        y_preds = []
        y_trues = []
        for row in scores:
            name = row["name"]
            values = [preds[c][name] for c in combo]
            votes = Counter(values).most_common(1)
            winner = votes[0][0]
            if winner and votes[0][1] >= args.vote_threshold:
                y_preds.append(float(winner))
                y_trues.append(trues[name])

        cells = metrics.confusion_matrix(y_trues, y_preds).ravel()
        if len(cells) < 4:
            continue

        tn, fp, fn, tp = cells

        bests.append(
            {
                "combo": combo,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "count": len(y_trues),
                "fraction": len(y_trues) / len(scores),
                "accuracy": metrics.accuracy_score(y_trues, y_preds),
                "f1": metrics.f1_score(y_trues, y_preds),
                "f0.5": metrics.fbeta_score(y_trues, y_preds, beta=0.5),
                "precision": metrics.precision_score(y_trues, y_preds),
                "recall": metrics.recall_score(y_trues, y_preds),
            }
        )

    print_summary(args, bests, metric)


def print_summary(args, bests, metric):
    # Filter combinations to remove really bad ones
    bests = [b for b in bests if b[args.combo_filter] >= args.combo_limit]
    # Now sort combos to find the best
    bests = sorted(bests, key=lambda m: m[args.combo_order], reverse=True)

    pp(bests[0]["combo"])
    print()
    for idx in util.METRICS_INT:
        print(idx.ljust(15), bests[0].get(idx, ""))
    for idx in util.METRICS_FLOAT:
        print(idx.ljust(15), f"{bests[0].get(idx, 0.0):0.3f}")

    for cp in bests[0]["combo"]:
        print()
        print(cp)
        for idx in util.METRICS_INT:
            print(idx.ljust(15), metric[idx][cp])
        for idx in util.METRICS_FLOAT:
            print(idx.ljust(15), f"{metric[idx][cp]:0.3f}")

    print()
    print("Top 10 combo scores")
    print()
    for i, combo in enumerate(bests[:10], 1):
        print(i, args.combo_order, f"{combo[args.combo_order]:0.3f}")
        pp(combo["combo"])
        print()

    log.finished()


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
        "--metric-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""File that contains metrics for the scores above.""",
    )

    arg_parser.add_argument(
        "--checkpoint-metric",
        choices=util.METRICS,
        default="accuracy",
        help="""Reject checkpoints with a metric below this value.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--checkpoint-limit",
        type=float,
        default=0.85,
        metavar="FRACTION",
        help="""Reject checkpoints with a metric below this value.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--votes",
        type=int,
        default=3,
        metavar="INT",
        help="""Vote this many times on each score. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--vote-threshold",
        type=int,
        default=2,
        metavar="INT",
        help="""Only keep predictions if they have this many votes.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--combo-order",
        choices=util.METRICS,
        default="accuracy",
        help="""Find the combo that maximizes this metric. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--combo-filter",
        choices=util.METRICS,
        default="accuracy",
        help="""Remove combos where this metric is to low. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--combo-limit",
        type=float,
        default=0.9,
        help="""Reject combos with a combined metric below this value.
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
