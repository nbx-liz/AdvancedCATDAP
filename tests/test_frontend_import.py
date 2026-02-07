import importlib


def test_dash_app_import_smoke():
    """Smoke test: dash_app can be imported without syntax/import errors."""
    mod = importlib.import_module("advanced_catdap.frontend.dash_app")
    assert hasattr(mod, "app")
