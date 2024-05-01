import csv
from pathlib import Path

import datasets
from PIL import Image

_CITATION = None

_DESCRIPTION = """Pre-training data for a Masked AutoEncoder."""

_HOMEPAGE = None

_LICENSE = None

_URLS = {
    "holdout": "mae_splits_224/holdout.csv",
    "train": "mae_splits_224/train.csv",
    "validation": "mae_splits_224/validation.csv",
}


class MaeSplits224(datasets.GeneratorBasedBuilder):
    """Pre-training data for a Masked AutoEncoder."""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file_name": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        filepath = Path(data_dir) / "mae_splits.csv"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with filepath.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                path = Path() / split / row["file_name"]
                image = Image.open(path)
                yield row["file_name"], {"pixel_values": image}
