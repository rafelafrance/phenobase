from pathlib import Path

from torch.utils.data import Dataset


class LabeledTraits(Dataset):
    def __init__(
        self,
        *,
        sheets: list[dict],
        labels: list[str],
        image_dir: Path,
        augment: bool = False,
    ) -> None:
        self.sheets = sheets
        self.labels = labels
        self.transforms = self.build_transforms(augment)

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        pass

    @staticmethod
    def build_transforms(*, augment=False):
        return []
