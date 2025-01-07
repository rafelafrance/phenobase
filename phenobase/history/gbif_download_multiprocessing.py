#!/usr/bin/env python3

from multiprocessing import Pool

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

        parallel_download(
            subdir, row_chunk, args.processes, args.attempts, args.max_width
        )

    log.finished()


def parallel_download(subdir, rows, processes, attempts, max_width):
    with Pool(processes=processes) as pool:
        results = [
            pool.apply_async(
                gbif_download.download,
                (
                    row["gbifID"],
                    row["tiebreaker"],
                    row["identifier"],
                    subdir,
                    attempts,
                    max_width,
                ),
            )
            for row in rows
        ]
        results = [r.get() for r in results]
    return results


if __name__ == "__main__":
    main()
