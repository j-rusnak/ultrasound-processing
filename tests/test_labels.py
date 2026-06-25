from pathlib import Path

from ultrasound_processing.data.labels import LabelRecord, load_labels


def test_load_labels_maps_study_ids_to_records(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Study_ID,FM,FFM,Weight_visit,Length_visit",
                "101,1.5,2.5,3.5,4.5",
                "202,6.0,7.0,8.0,9.0",
            ]
        ),
        encoding="utf-8",
    )

    labels = load_labels(csv_path)

    assert labels[101] == LabelRecord(fm=1.5, ffm=2.5, weight=3.5, length=4.5)
    assert labels[202].ffm == 7.0
