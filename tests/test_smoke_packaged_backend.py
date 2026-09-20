import pytest

from tools import smoke_packaged_backend


def test_benchmark_mode_separates_inference_and_roundtrip(monkeypatch):
    ticks = iter((1.000, 1.010, 2.000, 2.020))
    request_count = 0

    def fake_request_prediction(_socket, _events, _mode):
        nonlocal request_count
        request_count += 1
        return {"inference_ms": 8.0 if request_count == 6 else 12.0}

    monkeypatch.setattr(smoke_packaged_backend, "configure_mode", lambda *_: None)
    monkeypatch.setattr(
        smoke_packaged_backend,
        "request_prediction",
        fake_request_prediction,
    )
    monkeypatch.setattr(smoke_packaged_backend.time, "perf_counter", lambda: next(ticks))

    result = smoke_packaged_backend.benchmark_mode(
        object(),
        object(),
        "ellipse",
        repeats=2,
    )

    assert request_count == 7
    assert result["roundtrip_mean_ms"] == pytest.approx(15.0)
    assert result["roundtrip_p50_ms"] == pytest.approx(15.0)
    assert result["roundtrip_p95_ms"] == pytest.approx(19.5)
    assert result["inference_mean_ms"] == pytest.approx(10.0)
    assert result["inference_p50_ms"] == pytest.approx(10.0)
    assert result["inference_p95_ms"] == pytest.approx(11.8)
    assert result["other_mean_ms"] == pytest.approx(5.0)
