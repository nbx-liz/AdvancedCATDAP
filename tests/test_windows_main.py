from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from advanced_catdap.runtime import cli as runtime_cli
from advanced_catdap.runtime import gui as runtime_gui


def test_parse_args_gui_defaults():
    args = runtime_cli.parse_args([])
    assert args.mode == "desktop"
    assert args.db_path == "data/jobs.db"
    assert args.data_dir == "data"


def test_parse_args_worker_requires_fields():
    with pytest.raises(SystemExit):
        runtime_cli.parse_args(["--worker", "--job-id", "j1"])


def test_parse_args_supports_subcommand_worker():
    args = runtime_cli.parse_args(
        [
            "worker",
            "--job-id",
            "j1",
            "--dataset-id",
            "d1",
            "--params",
            "{}",
        ]
    )
    assert args.mode == "worker"
    assert args.job_id == "j1"


def test_main_worker_dispatch(monkeypatch):
    called = {}

    def _fake_run_worker_mode(args):
        called["job_id"] = args.job_id
        called["dataset_id"] = args.dataset_id
        called["params"] = args.params
        called["db_path"] = args.db_path
        called["data_dir"] = args.data_dir
        return 0

    monkeypatch.setattr(runtime_cli, "run_worker_mode", _fake_run_worker_mode)
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: 99)
    monkeypatch.setattr(runtime_cli, "run_web_mode", lambda: 98)

    exit_code = runtime_cli.main(
        [
            "--worker",
            "--job-id",
            "job123",
            "--dataset-id",
            "ds123",
            "--params",
            '{"target_col":"y"}',
            "--db-path",
            "tmp/jobs.db",
            "--data-dir",
            "tmp",
        ]
    )
    assert exit_code == 0
    assert called == {
        "job_id": "job123",
        "dataset_id": "ds123",
        "params": '{"target_col":"y"}',
        "db_path": "tmp/jobs.db",
        "data_dir": "tmp",
    }


def test_main_gui_path(monkeypatch):
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: 7)
    exit_code = runtime_cli.main([])
    assert exit_code == 7


def test_worker_mode_does_not_call_gui(monkeypatch):
    called = {"worker": False}

    def _fake_worker(_args):
        called["worker"] = True
        return 0

    monkeypatch.setattr(runtime_cli, "run_worker_mode", _fake_worker)
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: (_ for _ in ()).throw(RuntimeError("gui-called")))
    monkeypatch.setattr(runtime_cli, "run_web_mode", lambda: (_ for _ in ()).throw(RuntimeError("web-called")))
    code = runtime_cli.main(
        [
            "--worker",
            "--job-id",
            "j1",
            "--dataset-id",
            "d1",
            "--params",
            "{}",
        ]
    )
    assert code == 0
    assert called["worker"] is True


def test_main_gui_sets_desktop_mode(monkeypatch):
    monkeypatch.delenv("CATDAP_DESKTOP_MODE", raising=False)
    monkeypatch.setattr(runtime_gui, "find_free_port", lambda: 18080)
    monkeypatch.setattr(runtime_gui, "wait_for_server", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runtime_gui, "run_api_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_gui, "run_dash_server", lambda *_args, **_kwargs: None)

    class _FakeThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(runtime_gui.threading, "Thread", _FakeThread)

    calls = {"create_window": 0, "start": 0}
    fake_webview = SimpleNamespace()

    def _create_window(*_args, **_kwargs):
        calls["create_window"] += 1

    def _start(*_args, **_kwargs):
        calls["start"] += 1

    fake_webview.create_window = _create_window
    fake_webview.start = _start
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    code = runtime_gui.run_desktop_mode()
    assert code == 0
    assert calls["create_window"] == 1
    assert calls["start"] == 1
    assert runtime_gui.os.environ.get("CATDAP_DESKTOP_MODE") == "1"


def test_windows_main_script_is_thin_wrapper():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "windows_main.py"
    spec = spec_from_file_location("windows_main_module_wrapper", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is runtime_cli.main
