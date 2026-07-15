from threading import RLock

from backend.event_processing import (
    NOISE_FILTER_DISPLAY_NAMES,
    normalize_noise_filter_type,
    to_event_cd,
)


try:
    from metavision_sdk_cv import (
        ActivityNoiseFilterAlgorithm,
        AntiFlickerAlgorithm,
        SpatioTemporalContrastAlgorithm,
        TrailFilterAlgorithm,
    )
    _METAVISION_CV_IMPORT_ERROR = None
except Exception as exc:
    ActivityNoiseFilterAlgorithm = None
    AntiFlickerAlgorithm = None
    SpatioTemporalContrastAlgorithm = None
    TrailFilterAlgorithm = None
    _METAVISION_CV_IMPORT_ERROR = exc

try:
    from metavision_sdk_cv import FrequencyEstimationConfig
except Exception:
    FrequencyEstimationConfig = None


class NoiseFilterPipeline:
    def __init__(self, filter_type="none", threshold_us=10000, status_callback=None, report_initial_status=True):
        self.filter_type = normalize_noise_filter_type(filter_type)
        self.threshold_us = max(1, int(threshold_us or 10000))
        self.status_callback = status_callback
        self.report_initial_status = bool(report_initial_status)
        self.algorithm = None
        self.output = None
        self.warning_printed = False
        self.width = None
        self.height = None
        self._lock = RLock()

    @property
    def enabled(self):
        with self._lock:
            return self.algorithm is not None

    def initialize(self, width, height):
        with self._lock:
            self.width = int(width)
            self.height = int(height)
            self._initialize_locked(report_initial=True)

    def update_settings(self, filter_type, threshold_us):
        next_filter_type = normalize_noise_filter_type(filter_type)
        next_threshold_us = max(1, int(threshold_us or 10000))
        with self._lock:
            if next_filter_type == self.filter_type and next_threshold_us == self.threshold_us:
                return False
            self.filter_type = next_filter_type
            self.threshold_us = next_threshold_us
            self.algorithm = None
            self.output = None
            self.warning_printed = False
            if self.width is not None and self.height is not None:
                self._initialize_locked(report_initial=False)
        return True

    def reset(self):
        with self._lock:
            self.algorithm = None
            self.output = None
            self.warning_printed = False
            if self.width is not None and self.height is not None:
                self._initialize_locked(report_initial=False, report=False)

    def _initialize_locked(self, report_initial, report=True):
        self.algorithm = None
        self.output = None
        if self.filter_type == "none":
            if report:
                self._report_status("[NoiseFilter] Disabled", report_initial)
            return

        if _METAVISION_CV_IMPORT_ERROR is not None:
            if report:
                self._report_status(
                    f"[NoiseFilter] Metavision CV unavailable: {_METAVISION_CV_IMPORT_ERROR}",
                    report_initial,
                )
            return

        try:
            self.algorithm = self._create_algorithm(self.width, self.height)
            self.output = self.algorithm.get_empty_output_buffer()
            if report:
                self._report_status(
                    "[NoiseFilter] Enabled "
                    f"{NOISE_FILTER_DISPLAY_NAMES[self.filter_type]} "
                    f"(threshold={self.threshold_us}us)",
                    report_initial,
                )
        except Exception as exc:
            if report:
                self._report_status(
                    f"[NoiseFilter] Failed to initialize {self.filter_type}: {exc}",
                    report_initial,
                )
            self.algorithm = None
            self.output = None

    def apply(self, events):
        with self._lock:
            if self.algorithm is None or events is None or len(events) == 0:
                return events

            try:
                event_cd = to_event_cd(events)
                self.algorithm.process_events(event_cd, self.output)
                try:
                    return self.output.numpy(copy=True)
                except TypeError:
                    return self.output.numpy().copy()
            except Exception as exc:
                if not self.warning_printed:
                    self._report(f"[NoiseFilter] Filtering failed, passing raw events through: {exc}")
                    self.warning_printed = True
                return events

    def _create_algorithm(self, width, height):
        if self.filter_type == "activity":
            return ActivityNoiseFilterAlgorithm(width, height, self.threshold_us)
        if self.filter_type == "trail":
            return TrailFilterAlgorithm(width, height, self.threshold_us)
        if self.filter_type == "stc":
            return SpatioTemporalContrastAlgorithm(width, height, self.threshold_us, True)
        if self.filter_type == "anti_flicker":
            return self._create_anti_flicker_filter(width, height)

        self.filter_type = "none"
        raise ValueError("Unsupported noise filter type")

    def _create_anti_flicker_filter(self, width, height):
        if AntiFlickerAlgorithm is None:
            raise RuntimeError("AntiFlickerAlgorithm is unavailable")

        if FrequencyEstimationConfig is not None:
            flicker_config = FrequencyEstimationConfig()
            for attr, value in (
                ("filter_length", 7),
                ("min_freq", 50.0),
                ("max_freq", 70.0),
                ("diff_thresh_us", self.threshold_us),
                ("threshold", self.threshold_us),
            ):
                if hasattr(flicker_config, attr):
                    setattr(flicker_config, attr, value)
            try:
                return AntiFlickerAlgorithm(width, height, flicker_config)
            except TypeError:
                pass

        constructor_attempts = (
            (width, height, 7, 50.0, 70.0, self.threshold_us),
            (width, height, 50.0, 70.0, self.threshold_us),
            (width, height, self.threshold_us),
        )
        last_error = None
        for args in constructor_attempts:
            try:
                return AntiFlickerAlgorithm(*args)
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"AntiFlicker constructor is not compatible: {last_error}")

    def _report(self, message):
        if self.status_callback is not None:
            self.status_callback(message)

    def _report_initial(self, message):
        if self.report_initial_status:
            self._report(message)

    def _report_status(self, message, initial):
        if initial:
            self._report_initial(message)
        else:
            self._report(message)
