import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sample Excluded Herbarium Sheets

    Some herbarium sheets had inferences of flowers present/absent but were excluded from the final data set because of missing data in the GBIF metadata:
    - Missing collection date
    - Missing georeferences
    - Or both

    This notebook will sample the excluded sheets to see if there are any themes as to why they are missing the data.
    - Sample 500 records that have a winning presence/absence for each of the categories above.
    - Include the GBIF metadata for the sheet.
    - Include the full sized image of the herbarium sheet.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import csv
    import sqlite3
    from dataclasses import dataclass
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm
    return Path, pd


@app.cell
def _(Path):
    inference_path = Path("data") / "inference"

    winners_csv = inference_path / "flower_inference_votes_2025-09-02.csv"
    gbif_db = Path("data") / "gbif_2024-10-28.sqlite"

    hold_out_csv = inference_path / "ensemble_votes_holdout_2025-10-23.csv"
    return (hold_out_csv,)


@app.cell
def _(Path):
    excluded_path = Path("data") / "excluded"

    no_date_dir = excluded_path / "no_date"
    no_georef_dir = excluded_path / "no_georef"
    no_both_dir = excluded_path / "no_date_georef"

    no_date_csv = excluded_path / "no_date.csv"
    no_georef_csv = excluded_path / "no_georef.csv"
    no_both_csv = excluded_path / "no_date_georef.csv"
    return


@app.cell
def _(hold_out_csv, pd):
    df = pd.read_csv(hold_out_csv)
    return (df,)


@app.cell
def _(df):
    df2p = df.loc[(df["winner"].isna() & (df["expected"] == 1.0)), "expected"]
    df2n = df.loc[(df["winner"].isna() & (df["expected"] == 0.0)), "expected"]
    print(df2p.count(), df2n.count())
    return


@app.cell
def _(df):
    tp = df.loc[(df["winner"] == 1.0) & (df["expected"] == 1.0), "expected"]
    fp = df.loc[(df["winner"] == 1.0) & (df["expected"] == 0.0), "expected"]
    fn = df.loc[(df["winner"] == 0.0) & (df["expected"] == 1.0), "expected"]
    tn = df.loc[(df["winner"] == 0.0) & (df["expected"] == 0.0), "expected"]
    print(tp.count(), fp.count(), fn.count(), tn.count())
    total = tp.count() + fp.count() + fn.count() + tn.count()
    print(total)
    print(1442 - total)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
