#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
import warnings
from pathlib import Path

from PIL import Image
from pylib import gbif, inference, log, util
from tqdm import tqdm


def main(args):
    log.started(args=args)

    total, too_small, errors, channels, skipped, processed = 0, 0, 0, 0, 0, 0

    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        select = """
            select gbifID, tiebreaker, state
            from multimedia join occurrence using (gbifID)"""
        update = "update multimedia set state = ? where gbifid = ? and tiebreaker = ?"
        cursor = cxn.cursor()
        cursor.execute(select)

        while True:
            rows = cursor.fetchmany(inference.BATCH)
            if not rows:
                break

            total += len(rows)

            for row in tqdm(rows, desc=f"total: {total:,}"):
                rec: gbif.GbifRec = gbif.GbifRec(row)

                if not rec.good_image:
                    skipped += 1
                    continue

                path = rec.get_path(args.image_dir, debug=args.debug)

                if not path.exists():
                    skipped += 1
                    continue

                processed += 1

                if path.stat().st_size < util.TOO_DAMN_SMALL:
                    logging.warning(f"Image is too damn small {path.stem}")
                    too_small += 1
                    rec.state += " small"
                    cxn.execute(update, (rec.state, rec.gbifid, rec.tiebreaker))
                    continue

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=UserWarning
                    )  # No EXIF warnings

                    try:
                        with Image.open(path) as image:
                            if image.mode != "RGB":
                                print(f"RGB {path}")
                                image = image.convert("RGB")
                                image.save(path)
                                channels += 1

                    except util.IMAGE_ERRORS as err:
                        logging.warning(f"Image error {path.stem} {err}")
                        rec.state += " error"
                        errors += 1
                        cxn.execute(update, (rec.state, rec.gbifid, rec.tiebreaker))
                        continue

            cxn.commit()

    logging.info(f"Total records     {total:,}")
    logging.info(f"Skipped records   {skipped:,}")
    logging.info(f"Processed records {processed:,}")
    logging.info(f"Too small count   {too_small:,}")
    logging.info(f"Channel count     {channels:,}")
    logging.info(f"Error count       {errors:,}")
    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Fix images with incorrect channels or size."""),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This SQLite DB data contains image links.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Directory containing subdirectories with images.""",
    )

    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="""Use when debugging locally.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
