"""Optional INI30 annotation overlay, isolated from playback and inference."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen


ANNOTATION_FILE_NAME = "annotations.csv"
EVENT_FILE_NAME = "events.aedat4"
REQUIRED_COLUMNS = {
    "timestamp",
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "angle",
}


@dataclass(frozen=True)
class Ini30Ellipse:
    timestamp: int
    center_x: float
    center_y: float
    axis_x: float
    axis_y: float
    angle_degrees: float


class Ini30AnnotationIndex:
    """Timestamp index for one INI30 subject's ellipse annotations."""

    def __init__(self, samples, match_tolerance_us=None):
        self.samples = tuple(sorted(samples, key=lambda sample: sample.timestamp))
        self.timestamps = tuple(sample.timestamp for sample in self.samples)
        self.match_tolerance_us = (
            _default_match_tolerance(self.timestamps)
            if match_tolerance_us is None
            else max(0, int(match_tolerance_us))
        )

    @classmethod
    def from_csv(cls, csv_path):
        samples = []
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or ()))
                raise ValueError(f"INI30 annotations are missing columns: {', '.join(missing)}")

            for row in reader:
                sample = _parse_visible_sample(row)
                if sample is not None:
                    samples.append(sample)

        if not samples:
            raise ValueError("INI30 annotations do not contain visible ellipse samples")
        return cls(samples)

    def match(self, frame_timestamp):
        if not self.samples:
            return None
        timestamp = int(frame_timestamp)
        insertion = bisect.bisect_left(self.timestamps, timestamp)
        candidates = []
        if insertion < len(self.samples):
            candidates.append(self.samples[insertion])
        if insertion > 0:
            candidates.append(self.samples[insertion - 1])
        nearest = min(candidates, key=lambda sample: abs(sample.timestamp - timestamp))
        if abs(nearest.timestamp - timestamp) > self.match_tolerance_us:
            return None
        return nearest


class Ini30GroundTruthOverlay:
    """Self-contained, removable frame overlay for INI30 ground truth."""

    def __init__(self):
        self.enabled = False
        self.annotation_path = None
        self._index = None

    @property
    def available(self):
        return self._index is not None

    def configure_source(self, event_path):
        self.enabled = False
        self.annotation_path = None
        self._index = None

        annotation_path = find_ini30_annotations(event_path)
        if annotation_path is None:
            return False
        try:
            index = Ini30AnnotationIndex.from_csv(annotation_path)
        except (OSError, ValueError, TypeError):
            return False

        self.annotation_path = annotation_path
        self._index = index
        return True

    def set_enabled(self, enabled):
        self.enabled = bool(enabled) and self.available
        return self.enabled

    def draw(self, q_img, frame_timestamp, width, height):
        if not self.enabled or self._index is None:
            return False
        sample = self._index.match(frame_timestamp)
        if sample is None or not _ellipse_is_in_frame(sample, width, height):
            return False

        painter = QPainter(q_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(0, 255, 80))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 255, 80, 28))
        painter.save()
        painter.translate(sample.center_x, sample.center_y)
        painter.rotate(sample.angle_degrees)
        painter.drawEllipse(QPointF(0, 0), sample.axis_x, sample.axis_y)
        painter.restore()
        painter.end()
        return True


def find_ini30_annotations(event_path):
    if not event_path:
        return None
    source_path = Path(event_path)
    if source_path.name.lower() != EVENT_FILE_NAME:
        return None
    annotation_path = source_path.with_name(ANNOTATION_FILE_NAME)
    return annotation_path if annotation_path.is_file() else None


def _parse_visible_sample(row):
    visibility = str(row.get("possible", "visible")).strip().lower()
    if visibility and visibility not in {"visible", "true", "1", "yes"}:
        return None
    try:
        sample = Ini30Ellipse(
            timestamp=int(row["timestamp"]),
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            axis_x=float(row["axis_x"]),
            axis_y=float(row["axis_y"]),
            angle_degrees=float(row["angle"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    numeric_values = (
        sample.center_x,
        sample.center_y,
        sample.axis_x,
        sample.axis_y,
        sample.angle_degrees,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        return None
    if sample.axis_x <= 0 or sample.axis_y <= 0:
        return None
    return sample


def _default_match_tolerance(timestamps):
    gaps = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if 0 < current - previous <= 1_000_000
    ]
    if not gaps:
        return 150_000
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    return min(500_000, max(50_000, int(median_gap * 0.75)))


def _ellipse_is_in_frame(sample, width, height):
    return (
        int(width) > 0
        and int(height) > 0
        and -sample.axis_x < sample.center_x < int(width) + sample.axis_x
        and -sample.axis_y < sample.center_y < int(height) + sample.axis_y
    )
