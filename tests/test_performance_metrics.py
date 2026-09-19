from app.performance_metrics import PerformanceMetrics


def test_performance_metrics_reports_windowed_throughput_and_p95():
    metrics = PerformanceMetrics(window_s=5.0, clock=lambda: 0.0)
    metrics.record_frame(0.001, now=1.0)
    metrics.record_frame(0.003, now=2.0)
    metrics.record_prediction(now=2.0)

    snapshot = metrics.snapshot(now=5.0)

    assert snapshot.display_fps == 0.4
    assert snapshot.prediction_fps == 0.2
    assert snapshot.display_p95_ms == 3.0
    assert snapshot.has_activity


def test_performance_metrics_discards_samples_outside_window():
    metrics = PerformanceMetrics(window_s=5.0, clock=lambda: 0.0)
    metrics.record_frame(0.010, now=1.0)
    metrics.record_prediction(now=1.0)

    snapshot = metrics.snapshot(now=7.0)

    assert snapshot.frame_count == 0
    assert snapshot.prediction_count == 0
    assert not snapshot.has_activity
