import argparse

from advanced_catdap.runtime import cli as runtime_cli


def test_parse_args_web_subcommand():
    args = runtime_cli.parse_args(["web"])
    assert args.mode == "web"
    assert args.db_path == "data/jobs.db"
    assert args.data_dir == "data"


def test_parse_args_desktop_subcommand():
    args = runtime_cli.parse_args(["desktop"])
    assert args.mode == "desktop"


def test_parse_args_legacy_gui_alias():
    args = runtime_cli.parse_args(["gui"])
    assert args.mode == "desktop"


def test_main_dispatches_web(monkeypatch):
    monkeypatch.setattr(runtime_cli, "run_worker_mode", lambda _args: 99)
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: 98)
    monkeypatch.setattr(runtime_cli, "run_web_mode", lambda: 7)
    assert runtime_cli.main(["web"]) == 7


def test_main_dispatches_desktop(monkeypatch):
    monkeypatch.setattr(runtime_cli, "run_worker_mode", lambda _args: 99)
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: 6)
    monkeypatch.setattr(runtime_cli, "run_web_mode", lambda: 98)
    assert runtime_cli.main(["desktop"]) == 6


def test_legacy_worker_still_supported(monkeypatch):
    seen = {}

    def _fake_worker(args: argparse.Namespace) -> int:
        seen["job_id"] = args.job_id
        seen["dataset_id"] = args.dataset_id
        return 0

    monkeypatch.setattr(runtime_cli, "run_worker_mode", _fake_worker)
    monkeypatch.setattr(runtime_cli, "run_desktop_mode", lambda: 98)
    monkeypatch.setattr(runtime_cli, "run_web_mode", lambda: 97)

    code = runtime_cli.main(
        ["--worker", "--job-id", "j1", "--dataset-id", "d1", "--params", "{}"]
    )
    assert code == 0
    assert seen == {"job_id": "j1", "dataset_id": "d1"}

