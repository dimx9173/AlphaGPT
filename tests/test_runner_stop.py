"""Runner STOP-signal unit tests: file -> loop-exit semantics, no network."""


def _runner(tmp_path, monkeypatch):
    monkeypatch.setenv("STOP_SIGNAL_PATH", str(tmp_path / "STOP"))
    from strategy_manager import runner as R
    r = R.StrategyRunner.__new__(R.StrategyRunner)
    r.stop_signal_path = str(tmp_path / "STOP")
    return r


def test_no_file_means_no_stop(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    assert r._stop_requested() is False
    assert r._handle_stop_signal() is False


def test_stop_file_triggers_and_consumes(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    open(r.stop_signal_path, "w").write("STOP")
    assert r._stop_requested() is True
    assert r._handle_stop_signal() is True
    assert open(r.stop_signal_path).read() == "STOPPED"
    # consumed STOPPED still blocks
    assert r._stop_requested() is True


def test_arbitrary_content_does_not_stop(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    open(r.stop_signal_path, "w").write("GO")
    assert r._stop_requested() is False
    assert r._handle_stop_signal() is False
    assert open(r.stop_signal_path).read() == "GO"


def test_paper_mode_refuses_loop():
    import asyncio
    from strategy_manager import runner as R
    r = R.StrategyRunner.__new__(R.StrategyRunner)
    r.paper_mode = True
    asyncio.run(r.run_loop())
