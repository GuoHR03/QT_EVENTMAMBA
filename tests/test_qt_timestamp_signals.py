from backend.api import BackendAPI
from backend.Camera import CameraThread


INI30_TIMESTAMP = 1_650_973_019_494_464


def test_camera_thread_preserves_ini30_timestamp():
    received = []
    thread = CameraThread()
    thread.image_signal.connect(lambda _image, timestamp: received.append(timestamp))

    thread.image_signal.emit(None, INI30_TIMESTAMP)

    assert received == [INI30_TIMESTAMP]


def test_backend_api_preserves_frame_and_prediction_timestamps():
    api = BackendAPI()
    frames = []
    predictions = []
    api.image_signal.connect(lambda _image, timestamp: frames.append(timestamp))
    api.prediction_signal.connect(lambda _result, timestamp: predictions.append(timestamp))

    api.image_signal.emit(None, INI30_TIMESTAMP)
    api.prediction_signal.emit(None, INI30_TIMESTAMP)

    assert frames == [INI30_TIMESTAMP]
    assert predictions == [INI30_TIMESTAMP]
