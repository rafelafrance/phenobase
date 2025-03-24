#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from pathlib import Path

from pylib import log


def main(args):
    log.started()

    # count_gbif(args.gbif_db)
    count_angiosperm(args.angiosperm_db)

    log.finished()


def count_angiosperm(angiosperm_db):
    with sqlite3.connect(angiosperm_db) as cxn:
        sql = """select count(*) from angiosperms"""
        cursor = cxn.execute(sql)
        logging.info(f"Total angiosperm records:{cursor.fetchone()[0]:10,d}")

        prefix = "select count(*) from targets "

        sql = prefix + """where trait = 'flowering' and target = 1.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Flowers = present count: {cursor.fetchone()[0]:10,d}")

        sql = prefix + """where trait = 'flowering' and target = 0.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Flowers = absent count:  {cursor.fetchone()[0]:10,d}")

        sql = prefix + """where trait = 'fruiting' and target = 1.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Fruits = present count:  {cursor.fetchone()[0]:10,d}")

        sql = prefix + """where trait = 'fruiting' and target = 0.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Fruits = absent count:   {cursor.fetchone()[0]:10,d}")

        sql = prefix + """where trait = 'leaf_out' and target = 1.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Leaves = present count:  {cursor.fetchone()[0]:10,d}")

        sql = prefix + """where trait = 'leaf_out' and target = 0.0"""
        cursor = cxn.execute(sql)
        logging.info(f"Leaves = absent count:   {cursor.fetchone()[0]:10,d}")


def count_gbif(gbif_db):
    total = 0
    reproductivecondition = 0
    dynamicproperties = 0
    occurrenceremarks = 0
    fieldnotes = 0
    any_ = 0

    limit = 500_000
    offset = 0

    while True:
        with sqlite3.connect(gbif_db) as cxn:
            logging.info(f"Counting records [{offset:10,d} to {offset + limit:10,d}]")
            sql = f"""
                create temporary table images as
                select gbifid, reproductivecondition, dynamicproperties,
                    fieldnotes, occurrenceremarks
                from multimedia
                join occurrence using (gbifid)
                where state <> '' and state not like '%error'
                limit {limit} offset {offset}
            """
            cursor = cxn.execute(sql)

            sql = """select count(*) from images"""
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            if count == 0:
                break
            total += count

            sql = """select count(*) from images where reproductivecondition <> ''"""
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            reproductivecondition += count

            sql = """select count(*) from images where dynamicproperties <> ''"""
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            dynamicproperties += count

            sql = """select count(*) from images where occurrenceremarks <> ''"""
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            occurrenceremarks += count

            sql = """select count(*) from images where fieldnotes <> ''"""
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            fieldnotes += count

            sql = """
                select count(*) from images
                where reproductivecondition <> ''
                or dynamicproperties <> ''
                or occurrenceremarks <> ''
                or fieldnotes <> ''
                """
            cursor = cxn.execute(sql)
            count = cursor.fetchone()[0]
            any_ += count

        offset += limit

    logging.info(f"Total with images:     {total:10,d}")
    logging.info(f"reproductivecondition: {reproductivecondition:10,d}")
    logging.info(f"dynamicproperties:     {dynamicproperties:10,d}")
    logging.info(f"occurrenceremarks:     {occurrenceremarks:10,d}")
    logging.info(f"fieldnotes:            {fieldnotes:10,d}")
    logging.info(f"any:                   {any_:10,d}")


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Count which fields in the GBIF database have data."""
        ),
    )

    arg_parser.add_argument(
        "--gbif-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get data from this SQLite DB containing GBIF data.""",
    )

    arg_parser.add_argument(
        "--angiosperm-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get data from this SQLite DB containing angiosperm data.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
