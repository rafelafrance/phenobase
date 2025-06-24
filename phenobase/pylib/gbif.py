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
        self.tiebreaker = int(row["tiebreaker"])
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


def filter_bad_taxa(records: list[GbifRec], bad_taxa: list[tuple]) -> list[GbifRec]:
    if not bad_taxa:
        return records

    filtered = []
    for rec in records:
        if (rec.family, rec.genus) in bad_taxa or (rec.family, "") in bad_taxa:
            continue
        filtered.append(rec)
    return filtered


def get_bad_taxa(bad_taxa_csv: Path) -> list[tuple[str, str]]:
    if not bad_taxa_csv:
        return []

    bad_taxa = set()
    with bad_taxa_csv.open() as f:
        reader = csv.DictReader(f)
        bad_taxa = [
            (r["family"].lower(), r["genus"].lower() if r["genus"] else "")
            for r in reader
        ]

    return bad_taxa
