import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculate the confusion matrix for the holdout dataset""")
    return


@app.cell
def _():
    import csv
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import marimo as mo
    return Counter, Path, mo, pd


@app.cell
def _(mo):
    mo.md(r"""The name of the model checkpoints in the winning ensemble. The ensemble we used for final inference on 22 M records.""")
    return


@app.cell
def _():
    checkpoint_1 = "vit_384_lg_truth_f1_slurm_sl/checkpoint-9450"
    checkpoint_2 = "vit_384_lg_truth_f1_a/checkpoint-19199"
    checkpoint_3 = "effnet_528_truth_reg_f1_a/checkpoint-17424"
    checkpoints = [checkpoint_1, checkpoint_2, checkpoint_3]
    return checkpoint_1, checkpoint_2, checkpoint_3, checkpoints


@app.cell
def _(mo):
    mo.md(r"""Get the scores of the 3 winning model checkpoints on the holdout data.""")
    return


@app.cell
def _(Path, checkpoint_1, checkpoint_2, checkpoint_3):
    scores_dir = Path() / "data" / "scores"
    holdout_1_path = scores_dir / f"{checkpoint_1.replace('/', '-')}.csv"
    holdout_2_path = scores_dir / f"{checkpoint_2.replace('/', '-')}.csv"
    holdout_3_path = scores_dir / f"{checkpoint_3.replace('/', '-')}.csv"
    paths = [holdout_1_path, holdout_2_path, holdout_3_path]
    return holdout_1_path, holdout_2_path, holdout_3_path, scores_dir


@app.cell
def _(mo):
    mo.md(r"""These 2 CSV files hold the high and low thresholds for each model checkpoint in the winning ensemble, and yes the thresholds are the same, I checked.""")
    return


@app.cell
def _(scores_dir):
    ensemble_path = scores_dir / "best_3combo_fract.json"
    metrics_path = scores_dir / "flower_test_metrics.csv"
    return (metrics_path,)


@app.cell
def _(mo):
    mo.md(r"""Get the high and low thresholds for each model checkpoint in the winning ensemble.""")
    return


@app.cell
def _(checkpoints, metrics_path, pd):
    metrics = pd.read_csv(metrics_path).set_index("metric")
    lows = [metrics.loc["threshold_low", c] for c in checkpoints]
    highs = [metrics.loc["threshold_high", c] for c in checkpoints]
    return highs, lows


@app.cell
def _(mo):
    mo.md(r"""Read in the scores of the winning model checkpoints against the holdout dataset.""")
    return


@app.cell
def _(holdout_1_path, holdout_2_path, holdout_3_path, pd):
    holdout_1 = pd.read_csv(holdout_1_path)
    holdout_2 = pd.read_csv(holdout_2_path)
    holdout_3 = pd.read_csv(holdout_3_path)
    holdouts = [holdout_1, holdout_2, holdout_3]
    return holdout_1, holdout_2, holdout_3, holdouts


@app.cell
def _(mo):
    mo.md(r"""Each model has its own vote which is a score that is either above (>=) the high threshold or below (<) the low threshold. Otherwise the vote is None == equivocal for that particular model checkpoint.""")
    return


@app.cell
def _(highs, holdouts, lows):
    for i in range(3):
        col = f"pred_{i + 1}"
        holdouts[i][col] = None
        holdouts[i].loc[holdouts[i]["truth_score"] >= highs[i], col] = 1.0
        holdouts[i].loc[holdouts[i]["truth_score"] < lows[i], col] = 0.0
    return


@app.cell
def _(mo):
    mo.md(r"""Now build the dataframe that will have the final vote.""")
    return


@app.cell
def _(holdout_1, holdout_2, holdout_3, pd):
    vote_df = holdout_1["truth"]
    vote_df = pd.concat(
        [
            holdout_1["truth"],
            holdout_1["pred_1"],
            holdout_2["pred_2"],
            holdout_3["pred_3"],
        ],
        axis=1,
    )
    vote_df = vote_df.rename(columns={"truth": "truth"})
    vote_df["predicted"] = None
    vote_df
    return (vote_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Now I tally the votes from above. To declare a predicted for the herbarium sample there must be 2 or more votes for either presence (1) or absence (0). No votes (None or blank) do no contribute to the final predicted. So
    - 2 or more 1 votes indicate presence. A single 0 or None/blank are OK.
    - 2 or more 0 votes indicate absence. A single 1 or None/blank are OK.
    - The rest of the records are considered "equivocal".
    """
    )
    return


@app.cell
def _(Counter, vote_df):
    for j, row in vote_df.iterrows():
        tops = Counter([row["pred_1"], row["pred_2"], row["pred_3"]]).most_common()
        top = tops[0]
        if top[0] == 0.0 and top[1] >= 2:
            vote_df.loc[j, "predicted"] = 0
        elif top[0] == 1.0 and top[1] >= 2:
            vote_df.loc[j, "predicted"] = 1
    vote_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Now compute the predicted predicteds vs. the actual value determined by a subject matter expert. That is we are computing the confusion matrix for the ensemble predictions.
    - True positive (`tp`) = predicted positive and the true value is positive.
    - False positive (`fp`) = predicted positive and the true value is negative.
    - False negative (`fn`) = predicted negative and the true value is positive.
    - True negative (`tn`) = predicted negative and the true value is negative.
    """
    )
    return


@app.cell
def _(vote_df):
    tp = vote_df.loc[(vote_df["truth"] == 1) & (vote_df["predicted"] == 1)].shape[0]
    fp = vote_df.loc[(vote_df["truth"] == 0) & (vote_df["predicted"] == 1)].shape[0]
    fn = vote_df.loc[(vote_df["truth"] == 1) & (vote_df["predicted"] == 0)].shape[0]
    tn = vote_df.loc[(vote_df["truth"] == 0) & (vote_df["predicted"] == 0)].shape[0]
    tp, fp, fn, tn, (tp + fn + fp + tn)
    return


@app.cell
def _(mo):
    mo.md(r"""Finally, I look at the equivocal records and see how many the expert said were positive (`e_pos`) and negative (`e_neg`).""")
    return


@app.cell
def _(vote_df):
    equiv = vote_df["predicted"].isna().sum()

    e_pos = vote_df.loc[(vote_df["truth"] == 1) & (vote_df["predicted"].isna())].shape[0]

    e_neg = vote_df.loc[(vote_df["truth"] == 0) & (vote_df["predicted"].isna())].shape[0]

    equiv, e_pos, e_neg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
