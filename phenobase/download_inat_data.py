#!/usr/bin/env python
import argparse
import logging
import textwrap
import urllib.request
from collections import namedtuple
from datetime import date
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.error import URLError

from pylib import log

Download = namedtuple("Url", "url path")


def main():
    log.started()

    args = parse_args()

    today = date.today().isoformat() if args.today else ""

    targets = []

    if args.dwc_url:
        path = args.output_dir / Path(args.dwc_url).name
        path = path.with_stem(f"{path.stem}_{today}") if today else path
        targets.append(Download(args.dwc_url, path))

    if args.metadata_url:
        path = args.output_dir / Path(args.metadata_url).name
        path = path.with_stem(f"{path.stem}_{today}") if today else path
        targets.append(Download(args.metadata_url, path))

    threads = [Thread(target=downloader, args=(t,)) for t in targets]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    log.finished()


def downloader(target: Download):
    descr = f"{target.url} to {target.path}"

    logging.info(f"Downloading {descr}")

    try:
        urllib.request.urlretrieve(target.url, target.path)
    except (HTTPError, URLError) as err:
        logging.error(f"Could not download {descr} because of {err}")


def parse_args():
    description = """
        Download data from iNaturalist

        Download some big data files from iNaturalist.
        1. The most recent DwC archive of plant observations.
        2. The metadata file that contains image URLs.

        These files are large and will take a while to download.
        """

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--output-dir",
        metavar="PATH",
        required=True,
        type=Path,
        help="""Place downloaded files into this directory.""",
    )

    arg_parser.add_argument(
        "--dwc-url",
        metavar="URL",
        default="https://www.inaturalist.org/observations/phenobase-dwca.zip",
        help="""The URL for the most recent DwC archive of plant observations.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--metadata-url",
        metavar="URL",
        default=(
            "https://inaturalist-open-data.s3.amazonaws.com/metadata/"
            "inaturalist-open-data-latest.tar.gz"
        ),
        help="""The URL for the most recent metadata file for iNaturalist.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--today",
        action="store_true",
        help="""Append today's date to the downloaded file names.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
