import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from phenobase.pylib import const


@dataclass
class LabeledSheet:
    path: Path
    true: torch.tensor
    coreid: str


class LabeledDataset(Dataset):
    def __init__(
        self,
        data_csv: Path,
        split: const.SPLIT,
        trait: str,
        image_dir: Path,
        image_size: int,
        *,
        augment: bool = False,
        limit: int = 0,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.transform = self.build_transforms(image_size, augment=augment)

        self.sheets = []
        with data_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split or not row[trait] or row[trait] not in "01":
                    continue
                label = float(row[trait])
                self.sheets.append(
                    LabeledSheet(
                        path=image_dir / row["name"],
                        coreid=row["coreid"],
                        true=torch.tensor([label], dtype=torch.float),
                    )
                )
        self.sheets = self.sheets[:limit] if limit else self.sheets

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet.true, sheet.coreid

    @staticmethod
    def build_transforms(image_size, *, augment=False):
        xform = [v2.Resize((image_size, image_size))]

        if augment:
            xform += [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.AutoAugment(),
            ]

        xform += [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ConvertImageDtype(torch.float),
            v2.Normalize(const.IMAGENET_MEAN, const.IMAGENET_STD_DEV),
        ]

        return v2.Compose(xform)

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the trait."""
        pos = sum(s.true for s in self.sheets)
        pos_wt = (len(self) - pos) / pos if pos > 0.0 else 1.0
        return [pos_wt]
