from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script_module(name: str, relpath: str):
    script_path = Path(__file__).resolve().parents[1] / relpath
    spec = spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_desktop_on_windows_calls_desktop_mode(monkeypatch):
    module = _load_script_module("launch_desktop_module", "scripts/launch_desktop.py")
    calls = {}

    monkeypatch.setattr(module.platform, "system", lambda: "Windows")

    def _fake_main(argv):
        calls["argv"] = argv
        return 0

    monkeypatch.setattr(module, "main", _fake_main)
    code = module.launch()
    assert code == 0
    assert calls["argv"] == ["desktop"]

