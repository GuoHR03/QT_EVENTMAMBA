from backend.api import BackendAPI


class AppController:
    def __init__(self, settings):
        self.settings = settings
        self.backend = BackendAPI()
        self.input_file_path = None
        self.weights_path = None

    def is_camera_running(self):
        return self.backend.is_camera_running()

    def is_recording(self):
        return self.backend.is_recording()

    def is_inference_running(self):
        return self.backend.is_inference_running()

    def connect_view(self, image_handler, status_handler, prediction_handler, playback_finished_handler):
        self.backend.image_signal.connect(image_handler)
        self.backend.camera_status_signal.connect(status_handler)
        self.backend.prediction_signal.connect(prediction_handler)
        self.backend.playback_finished_signal.connect(playback_finished_handler)

    def sync_capture_settings(self, palette, fps):
        self.settings.update_capture(palette, fps)

    def start_camera(self):
        self.backend.start_camera(
            self.settings.palette,
            self.settings.fps,
            self.settings.roi,
        )

    def restart_camera_if_running(self):
        if not self.is_camera_running():
            return False
        self.backend.restart_camera(
            self.settings.palette,
            self.settings.fps,
            self.settings.roi,
        )
        return True

    def stop_camera(self):
        self.backend.stop_camera()

    def start_recording(self):
        self.backend.start_recording()

    def stop_recording(self):
        self.backend.stop_recording()

    def toggle_recording(self):
        if not self.is_camera_running():
            return None
        if self.is_recording():
            self.stop_recording()
            return False
        self.start_recording()
        return True

    def set_input_file(self, file_path, restart_if_running=False):
        self.input_file_path = file_path
        self.backend.set_input_file(file_path)
        if restart_if_running:
            return self.restart_camera_if_running()
        return False

    def set_weights_path(self, weights_path):
        self.weights_path = weights_path
        if self.is_inference_running():
            self.backend.stop_eventmamba()
            return True
        return False

    def load_model(self):
        if not self.weights_path:
            raise ValueError("weights_path is required")
        self.backend.start_eventmamba(self.weights_path)

    def unload_model(self):
        self.backend.stop_eventmamba()

    def apply_settings(self, roi, mode, filter_type, threshold_us):
        previous = (
            self.settings.roi,
            self.settings.noise_filter_type,
            self.settings.noise_filter_threshold_us,
        )

        self.settings.update_prediction(mode)
        self.backend.set_prediction_mode(mode)
        self.settings.update_noise_filter(filter_type, threshold_us)
        self.backend.set_noise_filter(filter_type, threshold_us, restart_camera=False)
        self.settings.update_roi(roi)

        current = (
            self.settings.roi,
            self.settings.noise_filter_type,
            self.settings.noise_filter_threshold_us,
        )
        changed = current != previous
        if changed:
            self.restart_camera_if_running()
        return changed

    def set_prediction_mode(self, mode):
        self.settings.update_prediction(mode)
        self.backend.set_prediction_mode(mode)

    def update_camera_roi(self, roi):
        self.settings.update_roi(roi)
        self.backend.update_camera_roi(self.settings.roi)

    def close(self):
        self.backend.close()
