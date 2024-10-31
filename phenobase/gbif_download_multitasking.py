#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor, as_completed

from pylib import gbif_download, log


def main():
    log.started()
    args = gbif_download.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)

    row_chunks = gbif_download.get_multimedia_recs(
        args.gbif_db, args.limit, args.offset
    )

    for row_chunk in row_chunks:
        subdir = gbif_download.mk_subdir(args.image_dir, args.dir_suffix)

        multitasking_download(
            subdir, row_chunk, args.max_workers, args.attempts, args.max_width
        )

    log.finished()


def multitasking_download(subdir, rows, max_workers, attempts, max_width):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                gbif_download.download,
                row["gbifID"],
                row["tiebreaker"],
                row["identifier"],
                subdir,
                attempts,
                max_width,
            )
            for row in rows
        ]
        results = list(as_completed(futures))
        return results


if __name__ == "__main__":
    main()
