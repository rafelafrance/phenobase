#!/usr/bin/env python3

import csv
from pathlib import Path

DIR = Path("data/images/phenobase")
# DIR = Path("datasets/images")
TRAIT = "flowers"
CSV = Path("datasets/training_data.csv")

with CSV.open() as f:
    reader = csv.DictReader(f)
    rows = [dict(r) for r in reader]

for r in rows:
    path = DIR / r["name"]
    if not path.exists():
        print(f"missing {r['name']}")
