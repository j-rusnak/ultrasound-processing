from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = ("Study_ID", "FM", "FFM", "Weight_visit", "Length_visit")


@dataclass(frozen=True)
class LabelRecord:
    fm: float
    ffm: float
    weight: float
    length: float

    def target(self, output: str) -> float:
        normalized = output.upper()
        if normalized == "FM":
            return self.fm
        if normalized == "FFM":
            return self.ffm
        raise ValueError("output must be 'FM' or 'FFM'")


def load_labels(path: str | Path) -> dict[int, LabelRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"labels CSV missing required column(s): {', '.join(missing)}")

        labels: dict[int, LabelRecord] = {}
        for row in reader:
            study_id = int(str(row["Study_ID"]).strip())
            labels[study_id] = LabelRecord(
                fm=float(row["FM"]),
                ffm=float(row["FFM"]),
                weight=float(row["Weight_visit"]),
                length=float(row["Length_visit"]),
            )
    return labels
