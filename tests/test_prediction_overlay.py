import sys
import types


# The default CI interpreter may have the PyQt6 package without loadable Qt
# DLLs.  tests/conftest.py provides the QtCore thread stubs used elsewhere;
# add only the tiny drawing surface needed to import prediction_overlay.
qtcore = sys.modules.get("PyQt6.QtCore")
qt_is_stubbed = qtcore is not None and not hasattr(qtcore, "__file__")
if qt_is_stubbed and not hasattr(qtcore, "QPointF"):
    qtcore.QPointF = lambda x, y: (x, y)

if qt_is_stubbed:
    pyqt6 = sys.modules.get("PyQt6") or types.ModuleType("PyQt6")
    qtgui = types.ModuleType("PyQt6.QtGui")
    qtgui.QColor = object
    qtgui.QPainter = object
    qtgui.QPen = object
    pyqt6.QtGui = qtgui
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtGui"] = qtgui
else:
    __import__("PyQt6.QtGui")


from app import prediction_overlay
from app.log_formatter import mode_display_name, roi_settings_message
from app.prediction_overlay import (
    PredictionSample,
    map_prediction_to_pixel,
    parse_prediction_result,
)
from backend.event_processing import normalize_roi


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PARTIAL_RAW_ROI = (-20, 600, 100, 200)
EFFECTIVE_ROI = (0, 600, 80, 120)


def test_cropped_prediction_maps_against_effective_intersection_of_raw_roi():
    sample = PredictionSample(
        values=(0.5, 0.5),
        coordinate_mode="norm",
        cropped=True,
    )

    assert normalize_roi(PARTIAL_RAW_ROI, FRAME_WIDTH, FRAME_HEIGHT) == EFFECTIVE_ROI
    assert map_prediction_to_pixel(
        sample,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        PARTIAL_RAW_ROI,
    ) == (40, 660)


def test_full_frame_normalized_one_maps_to_last_pixel_at_1280_by_720():
    sample = PredictionSample(
        values=(1.0, 1.0),
        coordinate_mode="norm",
        cropped=False,
    )

    assert map_prediction_to_pixel(
        sample,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        None,
    ) == (1279, 719)


def test_default_crop_normalized_one_stays_inside_crop_edge():
    sample = PredictionSample(
        values=(1.0, 1.0),
        coordinate_mode="norm",
        cropped=True,
    )

    assert map_prediction_to_pixel(sample, 640, 480, None) == (607, 479)


def test_prediction_effective_roi_snapshot_wins_over_current_roi():
    sample = parse_prediction_result(
        {
            "msg_type": "PREDICTION",
            "values": [0.5, 0.5],
            "cropped": True,
            "mode": "center",
            "effective_roi": PARTIAL_RAW_ROI,
        }
    )

    assert sample.has_effective_roi is True
    assert sample.effective_roi == PARTIAL_RAW_ROI
    assert map_prediction_to_pixel(
        sample,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        (800, 20, 200, 100),
    ) == (40, 660)


def test_draw_ellipse_scales_axes_to_effective_roi(monkeypatch):
    class FakePen:
        def __init__(self, _color):
            self.width = None

        def setWidth(self, width):
            self.width = width

    class FakePainter:
        def __init__(self):
            self.ellipse = None

        def setPen(self, _pen):
            pass

        def setBrush(self, _brush):
            pass

        def save(self):
            pass

        def translate(self, _x, _y):
            pass

        def rotate(self, _degrees):
            pass

        def drawEllipse(self, center, radius_x, radius_y):
            self.ellipse = (center, radius_x, radius_y)

        def restore(self):
            pass

        def end(self):
            pass

    painter = FakePainter()
    monkeypatch.setattr(prediction_overlay, "QPainter", lambda _image: painter)
    monkeypatch.setattr(prediction_overlay, "QColor", lambda *channels: channels)
    monkeypatch.setattr(prediction_overlay, "QPen", FakePen)
    monkeypatch.setattr(prediction_overlay, "QPointF", lambda x, y: (x, y))
    sample = PredictionSample(
        values=(0.5, 0.5, 0.25, 0.1, 0.0),
        coordinate_mode="norm",
        cropped=True,
        prediction_mode="ellipse",
        effective_roi=PARTIAL_RAW_ROI,
        has_effective_roi=True,
    )

    prediction_overlay.draw_prediction(
        object(),
        sample,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        current_roi=(800, 20, 200, 100),
    )

    assert painter.ellipse == ((0, 0), 20.0, 12.0)


def test_roi_settings_message_accepts_cleared_roi():
    message = roi_settings_message(None, "center")

    assert isinstance(message, str)
    assert message
    assert mode_display_name("center") in message
