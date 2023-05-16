#!/usr/bin/env python3
# aws s3 --no-sign-request --region us-east-1 cp
# s3://inaturalist-open-data/photos/37504/medium.jpg
# 37504_medium.jpg
# https://inaturalist-open-data.s3.amazonaws.com/photos/268542264/medium.jpg
import argparse
import textwrap
import time
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from pylib import log
from tqdm import tqdm

BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/{}.{}"
DELAY = 10  # Seconds to delay between attempts to download an image


def main():
    log.started()

    args = parse_args()

    df = pd.read_parquet(args.observations)

    image_ids = []
    suffixes = []

    grouped = df.groupby(["order", "flowering", "fruiting", "split"])
    for repro, group in grouped:
        group = group[: args.max_images]
        image_ids += group["photo_id"].tolist()
        suffixes += group["extension"].tolist()

    items = zip(image_ids, suffixes)

    results = {"exists": 0, "download": 0, "error": 0}

    with tqdm(total=len(image_ids)) as bar:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for image_id, suffix in items:
                futures.append(
                    executor.submit(
                        download,
                        image_id,
                        suffix,
                        args.image_dir,
                        args.image_size,
                        args.attempts,
                    )
                )
            for future in as_completed(futures):
                results[future.result()] += 1
                if args.print_results:
                    print(results)
                else:
                    bar.update(1)

    log.finished()


def download(image_id, suffix, image_dir, image_size, attempts):
    path = image_dir / f"{image_id}_{image_size}.{suffix}"
    url = BASE_URL.format(image_id, image_size, suffix)
    if path.exists():
        return "exists"
    for attempt in range(attempts):
        try:
            image = requests.get(url).content
            with open(path, "wb") as out_file:
                out_file.write(image)
            return "download"
        except (TimeoutError, ConnectionError):
            time.sleep(DELAY)
    return "error"


def parse_args():
    description = """Download images for observations."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--observations",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read from this parquet file containing observations.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        required=True,
        type=Path,
        help="""Place downloaded images into this directory.""",
    )

    arg_parser.add_argument(
        "--max-images",
        metavar="INT",
        type=int,
        default=1000,
        help="""How many images to download for each classification group.
            Groups are: phylogenetic order x phenology class (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-workers",
        metavar="INT",
        type=int,
        default=100,
        help="""How many worker threads to spawn. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--image-size",
        choices=["medium", "large", "original"],
        default="medium",
        help="""Download image size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--attempts",
        metavar="INT",
        type=int,
        default=3,
        help="""How many times to try downloading the image. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print result categories instead of a tqdm status bar.",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
