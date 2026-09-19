import csv

from app.ini30_ground_truth import (
    Ini30AnnotationIndex,
    Ini30Ellipse,
    find_ini30_annotations,
)


def test_finds_annotations_only_beside_ini30_event_file(tmp_path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("timestamp,center_x,center_y,axis_x,axis_y,angle\n", encoding="utf-8")

    assert find_ini30_annotations(tmp_path / "events.aedat4") == annotations
    assert find_ini30_annotations(tmp_path / "other.aedat4") is None
    assert find_ini30_annotations(None) is None


def test_annotation_index_matches_nearest_sample_with_bounded_age():
    first = Ini30Ellipse(1_000_000, 10, 20, 3, 4, 5)
    second = Ini30Ellipse(1_200_000, 11, 21, 3, 4, 6)
    index = Ini30AnnotationIndex((first, second))

    assert index.match(1_090_000) == first
    assert index.match(1_110_000) == second
    assert index.match(1_500_001) is None


def test_csv_loader_skips_non_visible_and_invalid_rows(tmp_path):
    csv_path = tmp_path / "annotations.csv"
    columns = [
        "timestamp", "center_x", "center_y", "axis_x", "axis_y", "angle", "possible"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(dict(zip(columns, (100, 10, 20, 3, 4, 30, "visible"))))
        writer.writerow(dict(zip(columns, (200, 11, 21, 3, 4, 31, "occluded"))))
        writer.writerow(dict(zip(columns, (300, 12, 22, -1, 4, 32, "visible"))))

    index = Ini30AnnotationIndex.from_csv(csv_path)

    assert len(index.samples) == 1
    assert index.samples[0].timestamp == 100
