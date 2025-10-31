import marimo

__generated_with = "0.17.4"
app = marimo.App(width="full", css_file="custom.css")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Sample Excluded Herbarium Sheets

    Some herbarium sheets had inferences of flowers present/absent but were excluded from the final gbif_data set because of missing gbif_data in the GBIF metagbif_data:
    - Missing collection date
    - Missing georeferences
    - Or both

    This notebook will sample the excluded sheets to see if there are any themes as to why they are missing the gbif_data.
    - Sample 500 records that have a winning presence/absence for each of the categories above.
    - Include the GBIF metagbif_data for the sheet.
    - Include the full sized image of the herbarium sheet.
    """)
    return


@app.cell
def _():
    import csv
    import random
    import sqlite3
    import time
    from pathlib import Path

    import pandas as pd
    return Path, csv, pd, random, sqlite3


@app.cell
def _(mo):
    mo.md(r"""
    The gbif_data sources we are using:
    - The votes CSV contains what the final "best" ensemble chose as as the result.
    - The GBIF gbif_database with all of the herbarium sheet metagbif_data.
    """)
    return


@app.cell
def _(Path):
    inference_path = Path("data") / "inference"

    winners_csv = inference_path / "flower_inference_votes_2025-09-02.csv"
    gbif_db = Path("data") / "gbif_2024-10-28.sqlite"
    return gbif_db, winners_csv


@app.cell
def _(mo):
    mo.md(r"""gbif_data sinks where we store the images and CSVs.""")
    return


@app.cell
def _(Path):
    excluded_dir = Path("data") / "excluded"

    no_date_dir = excluded_dir / "no_date"
    no_georef_dir = excluded_dir / "no_georef"
    no_both_dir = excluded_dir / "no_date_georef"
    return (excluded_dir,)


@app.cell
def _():
    download_limit = 500
    return (download_limit,)


@app.cell
def _():
    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """
    return (sql,)


@app.cell
def _():
    no_date_list = []
    no_georef_list = []
    no_both_list = []
    return no_both_list, no_date_list, no_georef_list


@app.cell
def _(
    csv,
    download_limit,
    gbif_db,
    no_both_list,
    no_date_list,
    no_georef_list,
    random,
    sql,
    sqlite3,
    winners_csv,
):
    def sample_recs() -> tuple[list[dict], list[dict], list[dict]]:
        with sqlite3.connect(gbif_db) as cxn, winners_csv.open() as win:
            cxn.row_factory = sqlite3.Row

            reader = csv.DictReader(win)
            all_winner_rows = [r for r in reader]
            random.shuffle(all_winner_rows)

            for i, winner_row in enumerate(all_winner_rows):
                if not (winner_row["winner"] and float(winner_row["winner"]) == 1.0):
                    continue

                gbif_id, tiebreaker = winner_row["id"].split("_")

                gbif_data = dict(cxn.execute(sql, (gbif_id, tiebreaker)).fetchone())

                event_date = bool(gbif_data["eventDate"])
                lat_long = bool(
                    gbif_data["decimalLatitude"] and gbif_data["decimalLatitude"]
                )

                if not event_date and lat_long and len(no_date_list) < download_limit:
                    no_date_list.append(winner_row | gbif_data)

                elif not lat_long and event_date  and len(no_georef_list) < download_limit:
                    no_georef_list.append(winner_row | gbif_data)

                elif not event_date and not lat_long and len(no_both_list) < download_limit:
                    no_both_list.append(winner_row | gbif_data)

                if (
                    len(no_date_list) >= download_limit
                    and len(no_georef_list) >= download_limit
                    and len(no_both_list) >= download_limit
                ):
                    return
    return (sample_recs,)


@app.cell
def _(sample_recs):
    sample_recs()
    return


@app.cell
def _(Path, excluded_dir):
    dst_dir = Path("/blue/guralnick/rafe.lafrance/phenobase/data/images/sample")
    src_dir = Path("/blue/guralnick/share/phenobase_specimen_data/images")

    no_date_bash = excluded_dir / "no_date.bash"
    no_georef_bash = excluded_dir / "no_georef.bash"
    no_both_bash = excluded_dir / "no_date_georef.bash"

    no_date_csv = excluded_dir / "no_date.csv"
    no_georef_csv = excluded_dir / "no_georef.csv"
    no_both_csv = excluded_dir / "no_date_georef.csv"
    return (
        dst_dir,
        no_both_bash,
        no_both_csv,
        no_date_bash,
        no_date_csv,
        no_georef_bash,
        no_georef_csv,
        src_dir,
    )


@app.cell
def _(Path, dst_dir, pd, src_dir):
    def write_files(records: list[dict], bash_file: Path, csv_file: Path, dst: Path) -> None:
        with bash_file.open("w") as fout:
            for rec in records:
                src = src_dir / rec["state"] / f"{rec['id']}.jpg"
                dst = dst_dir / dst
                fout.write(f"cp {src} {dst}\n")

        df = pd.DataFrame(records)
        df.to_csv(csv_file, index=False)
    return (write_files,)


@app.cell
def _(
    no_both_bash,
    no_both_csv,
    no_both_list,
    no_date_bash,
    no_date_csv,
    no_date_list,
    no_georef_bash,
    no_georef_csv,
    no_georef_list,
    write_files,
):
    write_files(no_date_list, no_date_bash, no_date_csv, "no_date")
    write_files(no_georef_list, no_georef_bash, no_georef_csv, "no_georef")
    write_files(no_both_list, no_both_bash, no_both_csv, "no_date_georef")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
