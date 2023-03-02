#!/usr/bin/env python
import argparse
import logging
import tarfile
import textwrap
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pylib import log


def main():
    log.started()

    args = parse_args()

    df = extract_dwc(args.dwc_file)
    df = merge_metadata(args.metadata_file, df)

    df.to_parquet(args.out_parquet, index=False)

    log.finished()


def extract_dwc(dwc_zip):
    logging.info("Extracting annotated observations from DwC zip file")

    file_name = "observations.csv"

    with ZipFile(dwc_zip) as zippy:
        df = pd.read_csv(zippy.open(file_name))

    df = df.loc[df["datasetName"].str.contains("research-grade", regex=False)]
    df = df.set_index("otherCatalogueNumbers")

    return df


def merge_metadata(metadata_tgz, dwc_df):
    file_name = "photos.csv"

    with tarfile.open(metadata_tgz, mode="r:gz") as tar:
        logging.info(f"Extracting {file_name} from the metadata tar gzip file")

        logging.info("  Find member name")
        target = next(n for n in tar.getnames() if n.endswith(file_name))

        logging.info("  Read in CSV file")
        columns = """ photo_id observation_uuid extension """.split()
        df = pd.read_csv(tar.extractfile(target), usecols=columns, sep="\t")

        logging.info("  Filter rows by observation_uuid")
        df = df.loc[df["observation_uuid"].isin(dwc_df.index)]

        logging.info("  Filter rows by count")
        counts = df.groupby("observation_uuid").count().reset_index()
        ids = counts.loc[counts["photo_id"] == 1, "observation_uuid"]
        df = df.loc[df["observation_uuid"].isin(ids)]

        logging.info("  Merge data frames")
        df = df.set_index("observation_uuid")
        df = df.join(dwc_df, how="inner")

    return df


def parse_args():
    description = """Filter the downloaded iNaturalist data and write it to a DB."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--dwc-file",
        metavar="PATH",
        type=Path,
        help="""Extract data from this DwC zip file.""",
    )

    arg_parser.add_argument(
        "--metadata-file",
        metavar="PATH",
        type=Path,
        required=True,
        help="""The iNaturalist metadata gzipped tar file.""",
    )

    arg_parser.add_argument(
        "--out-parquet",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Place extracted files into this directory.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
