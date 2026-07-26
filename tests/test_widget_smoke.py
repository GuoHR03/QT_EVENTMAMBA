from app.runtime_smoke import runtime_smoke_exit_code


def test_runtime_smoke_flag_runs_requested_check():
    calls = []

    exit_code = runtime_smoke_exit_code(
        ["UI_Event.exe", "--runtime-smoke-test"],
        runner=lambda: calls.append("smoke"),
    )

    assert exit_code == 0
    assert calls == ["smoke"]
