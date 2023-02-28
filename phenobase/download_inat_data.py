#!/usr/bin/env python
import argparse
import logging
import textwrap
import urllib.request
from collections import namedtuple
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.error import URLError

from pylib import log

Download = namedtuple("Url", "url path")


def main():
    log.started()

    args = parse_args()

    targets = []

    if args.dwc_url and args.dwc_file:
        targets.append(Download(args.dwc_url, args.dwc_file))

    if args.metadata_url and args.metadata_file:
        targets.append(Download(args.metadata_url, args.metadata_file))

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

        These files are large and will take
        """

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--dwc-url",
        metavar="URL",
        default="https://www.inaturalist.org/observations/phenobase-dwca.zip",
        help="""The URL for the most recent DwC archive of plant observations.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--dwc-file",
        metavar="PATH",
        type=Path,
        help="""Where to save the DwC archive zip file. If you don't enter this
            the DwC archive will not get downloaded.""",
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
        "--metadata-file",
        metavar="PATH",
        type=Path,
        help="""Where to save the metadata file. If you do not enter this the metadata
            file will not get downloaded.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
