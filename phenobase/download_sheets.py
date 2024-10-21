#!/usr/bin/env python3

# import argparse
# import textwrap
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
#
# import pandas as pd
# import requests
# from pylib import log
# from tqdm import tqdm
#
# DELAY = 10  # Seconds to delay between attempts to download an image
#
#
# def main():
#     log.started()
#     args = parse_args()
#
#     df = pd.read_parquet(args.parquet)
#     df = df[: args.limit]
#     df = df[:, ["gbifID", "identifier"]]
#     images = df.to_dict()
#
#     results = {"exists": 0, "download": 0, "error": 0}
#
#     with tqdm(total=len(images)) as bar:
#         with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#             futures = []
#             for image_id, suffix in items:
#                 futures.append(
#                     executor.submit(
#                         download,
#                         image_id,
#                         suffix,
#                         args.image_dir,
#                         args.image_size,
#                         args.attempts,
#                     )
#                 )
#             for future in as_completed(futures):
#                 results[future.result()] += 1
#                 if args.print_results:
#                     print(results)
#                 else:
#                     bar.update(1)
#     log.finished()
#
#
# def download(image_id, suffix, image_dir, image_size, attempts):
#     path = image_dir / f"{image_id}_{image_size}.{suffix}"
#     url = BASE_URL.format(image_id, image_size, suffix)
#     if path.exists():
#         return "exists"
#     for _attempt in range(attempts):
#         try:
#             image = requests.get(url, timeout=DELAY).content
#             with path.open("wb") as out_file:
#                 out_file.write(image)
#         except (TimeoutError, ConnectionError):
#             time.sleep(DELAY)
#         else:
#             return "download"
#     return "error"
#
#
# def parse_args():
#     description = """Download GBIF images."""
#
#     arg_parser = argparse.ArgumentParser(
#         description=textwrap.dedent(description), fromfile_prefix_chars="@"
#     )
#
#     arg_parser.add_argument(
#         "--raw-dir",
#         metavar="PATH",
#         required=True,
#         type=Path,
#         help="""Place downloaded images into this directory, before resizing.""",
#     )
#
#     arg_parser.add_argument(
#         "--resize-dir",
#         metavar="PATH",
#         required=True,
#         type=Path,
#         help="""Place resized images into this directory.""",
#     )
#
#     arg_parser.add_argument(
#         "--parquet",
#         metavar="PATH",
#         required=True,
#         type=Path,
#         help="""Parquet file containing the URLs of images.""",
#     )
#
#     arg_parser.add_argument(
#         "--resize",
#         type=int,
#         default=800,
#         help="""Resize images to use this as the width. The height will stay
#             proportional (default: %(default)s)""",
#     )
#
#     arg_parser.add_argument(
#         "--limit",
#         type=int,
#         help="""Limit to this many completed downloads.""",
#     )
#
#     arg_parser.add_argument(
#         "--attempts",
#         metavar="INT",
#         type=int,
#         default=3,
#         help="""How many times to try downloading the image.
#           (default: %(default)s)""",
#     )
#
#     args = arg_parser.parse_args()
#     return args
#
#
# if __name__ == "__main__":
#     main()
