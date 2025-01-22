import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from phenobase.pylib import const, image_util


@dataclass
class LabeledSheet:
    path: Path
    target: torch.tensor


class SingleLabelDataset(Dataset):
    def __init__(
        self,
        *,
        trait_csv: Path,
        image_dir: Path,
        split: const.SPLIT,
        image_size: int,
        trait: list[str],
        augment: bool = False,
    ) -> None:
        self.transform = image_util.build_transforms(image_size, augment=augment)
        self.trait = trait

        with trait_csv.open() as csv_in:
            reader = csv.DictReader(csv_in)
            self.sheets = [
                LabeledSheet(image_dir / s["name"], torch.tensor([float(s["value"])]))
                for s in reader
                if s["split"] == split
            ]

    def __len__(self) -> int:
        return len(self.sheets)

    def __getitem__(self, index) -> dict:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(sheet.path).convert("RGB")
            image = self.transform(image)
        return {"pixel_values": image, "labels": sheet.target, "name": sheet.path.name}

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the traits."""
        pos = sum(s.target[0] for s in self.sheets)
        neg = len(self) - pos
        pos_wt = neg / pos if pos > 0 else 1.0
        return torch.tensor(pos_wt, dtype=torch.float)
