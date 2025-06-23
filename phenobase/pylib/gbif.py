import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GbifRec:
    dir: str = ""
    stem: str = ""
    state: str = ""
    gbifid: str = ""
    family: str = ""
    genus: str = ""
    sci_name: str = ""
    tiebreaker: int = 0

    def __init__(self, row):
        row = dict(row)

        self.state = row["state"]
        self.tiebreaker = row["tiebreaker"]
        self.gbifid = row["gbifID"]

        self.family = row.get("family", "").lower()
        self.sci_name = row.get("scientificName", "").lower()

        self.genus = row.get("genus", "").lower()
        if not self.genus and self.sci_name:
            self.genus = self.sci_name.split()[0]

        parts = self.state.split()
        self.dir = parts[0]
        self.stem = f"{self.gbifid}_{self.tiebreaker}"
        self.stem += f"_{parts[-1]}" if len(parts) > 1 else ""

    @property
    def id(self):
        return self.stem

    @property
    def good_image(self):
        return self.state.startswith("images") and not (
            self.state.endswith("error") or self.state.endswith("small")
        )

    @property
    def bad_image(self):
        return self.state.endswith("error")

    @property
    def too_small(self):
        return self.state.endswith("small")

    @property
    def no_url(self):
        return not self.state.startswith("image")

    def get_path(self, image_dir, *, debug: bool = False):
        if debug:
            return self.local_path(image_dir)
        return self.hipergator_path(image_dir)

    def local_path(self, image_dir):
        return image_dir / (self.stem + ".jpg")

    def hipergator_path(self, image_dir):
        return image_dir / self.dir / (self.stem + ".jpg")

    def as_dict(self):
        return {"id": self.id, **asdict(self)}


def filter_bad_images(records: list[GbifRec]) -> list[GbifRec]:
    filtered = [r for r in records if r.good_image]
    return filtered


def filter_bad_taxa(
    records: list[GbifRec], bad_taxa: Path | None = None
) -> list[GbifRec]:
    if not bad_taxa:
        return records

    # Get the bad taxa
    to_remove = set()
    with bad_taxa.open() as f:
        reader = csv.DictReader(f)
        to_remove = [
            (r["family"].lower(), r["genus"].lower() if r["genus"] else "")
            for r in reader
        ]

    # Filter bad taxa from the rows
    filtered = []
    for rec in records:
        if (rec.family, rec.genus) in to_remove or (rec.family, "") in to_remove:
            continue
        filtered.append(rec)
    return filtered
