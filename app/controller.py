class AppController:
    def __init__(self, settings, backend=None):
        self.settings = settings
        if backend is None:
            from backend.api import BackendAPI

            backend = BackendAPI()
        self.backend = backend
        self.input_file_path = None
        self.weights_path = None

    def is_camera_running(self):
        return self.backend.is_camera_running()

    def is_recording(self):
        return self.backend.is_recording()

    def is_inference_running(self):
        return self.backend.is_inference_running()

    @property
    def inference_state(self):
        return self.backend.inference.state

    @property
    def inference_last_error(self):
        return self.backend.inference.last_error

    @property
    def source_mode(self):
        return self.backend.source_mode

    @property
    def inference_runtime_kind(self):
        return self.backend.inference.runtime_kind

    @property
    def inference_runtime_display_name(self):
        return self.backend.inference.runtime_display_name

    @property
    def active_model_path(self):
        return self.backend.inference.active_model_path

    def connect_view(self, image_handler, status_handler, prediction_handler, playback_finished_handler, progress_handler=None):
        self.backend.image_signal.connect(image_handler)
        self.backend.camera_status_signal.connect(status_handler)
        self.backend.prediction_signal.connect(prediction_handler)
        self.backend.playback_finished_signal.connect(playback_finished_handler)
        if progress_handler is not None:
            self.backend.playback_progress_signal.connect(progress_handler)

    def sync_capture_settings(self, palette, fps, replay_factor=1.0):
        self.settings.update_capture(palette, fps, replay_factor)

    def start_camera(self):
        self.backend.start_camera(self.settings.playback_config)

    def restart_camera_if_running(self):
        if not self.is_camera_running():
            return False
        self.backend.restart_camera(self.settings.playback_config)
        return True

    def update_replay_factor(self):
        self.backend.update_playback_config(self.settings.playback_config)

    def update_display_settings(self):
        self.backend.update_playback_config(self.settings.playback_config)

    def stop_camera(self):
        self.backend.stop_camera()

    def seek_playback(self, seek_fraction):
        self.backend.seek_playback(seek_fraction)

    def start_recording(self):
        return self.backend.start_recording()

    def stop_recording(self):
        return self.backend.stop_recording()

    def toggle_recording(self):
        if not self.is_camera_running():
            return None
        if self.is_recording():
            return False if self.stop_recording() else None
        return bool(self.start_recording())

    def set_input_file(self, file_path, restart_if_running=False):
        if not file_path:
            return self.set_live_camera()

        self.input_file_path = str(file_path)
        return self.backend.set_input_file(
            self.input_file_path,
            config=self.settings.playback_config,
            restart_if_running=restart_if_running,
        )

    def set_live_camera(self):
        self.input_file_path = None
        return self.backend.set_live_camera(config=self.settings.playback_config)

    def set_weights_path(self, weights_path):
        self.weights_path = weights_path
        return self.weights_path

    def load_model(self):
        """Start only the blocking inference backend phase.

        NetworkThread is a Qt object, so the view starts it separately on the
        UI thread after this worker operation succeeds.
        """
        if not self.weights_path:
            raise ValueError("weights_path is required")
        return self.backend.start_eventmamba_backend(self.weights_path)

    def start_model_network(self):
        """Create NetworkThread on the caller's (UI) thread."""
        return self.backend.start_eventmamba_network()

    def stop_model_network(self):
        """Destroy NetworkThread on the caller's (UI) thread."""
        return self.backend.stop_eventmamba_network()

    def unload_model(self):
        """Stop only the blocking inference backend phase."""
        return self.backend.stop_eventmamba_backend()

    def restart_model(self):
        if not self.weights_path:
            raise ValueError("weights_path is required")
        return self.backend.restart_eventmamba_backend()

    def cancel_model_start(self):
        return self.backend.cancel_eventmamba_start()

    def apply_settings(self, roi, mode, filter_type, threshold_us):
        previous = self.settings.playback_config

        self.settings.update_prediction(mode)
        self.backend.set_prediction_mode(mode)
        self.settings.update_noise_filter(filter_type, threshold_us)
        self.settings.update_roi(roi)

        current = self.settings.playback_config
        changed = current != previous
        if changed:
            self.backend.update_playback_config(current)
        return changed

    def close_ui_resources(self):
        """Release Qt-owned resources; call this from the UI thread."""
        errors = []
        try:
            self.stop_camera()
        except Exception as exc:
            errors.append(exc)
        try:
            self.stop_model_network()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))
        return True

    def close_backend_resources(self):
        """Release the blocking backend process; call this from a worker."""
        return self.unload_model()
