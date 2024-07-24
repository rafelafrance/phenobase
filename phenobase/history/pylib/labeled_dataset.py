import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from phenobase.pylib import util


@dataclass
class LabeledSheet:
    image_path: Path
    targets: torch.float


class LabeledDataset(Dataset):
    def __init__(
        self,
        csv_data,
        image_dir: Path,
        split: str,
        image_size: int,
        *,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = (image_size, image_size)
        self.transform = self.build_transforms(augment=augment)
        self.sheets: list[LabeledSheet] = []
        for sheet in csv_data:
            if sheet["split"] != split:
                continue
            targets = [self.str_to_target(sheet[t]) for t in util.TARGETS]
            self.sheets.append(
                LabeledSheet(
                    image_path=image_dir / sheet["file"],
                    targets=torch.tensor(targets, dtype=torch.float),
                )
            )

    @staticmethod
    def str_to_target(value):
        match value:
            case "1":
                return 1.0
            case "0":
                return 0.0
            case _:  # Guessing, if the expert can't tell then neither should the model?
                return 0.5

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        sheet: LabeledSheet = self.sheets[index]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            image = Image.open(sheet.image_path).convert("RGB")
            image = self.transform(image)

        return image, sheet.targets

    def build_transforms(self, *, augment=False):
        xform = [transforms.Resize(self.image_size)]

        if augment:
            xform += [
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        xform += [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(util.IMAGENET_MEAN, util.IMAGENET_STD_DEV),
        ]

        return transforms.Compose(xform)

    @staticmethod
    def pos_weight(dataset):
        weights = []
        for i in range(len(util.TARGETS)):
            pos = sum(s.targets[i] for s in dataset.sheets)
            pos_wt = (len(dataset) - pos) / pos if pos > 0.0 else 1.0
            weights.append(pos_wt)
        return weights
