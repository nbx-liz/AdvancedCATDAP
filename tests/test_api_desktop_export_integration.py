from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from advanced_catdap.service import api as api_module


@pytest.mark.integration
def test_api_desktop_export_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CATDAP_DESKTOP_MODE", "1")
    client = TestClient(api_module.app)
    payload = {
        "result": {
            "mode": "CLASSIFICATION",
            "baseline_score": 10.0,
            "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
            "interaction_importances": [],
            "feature_details": {},
            "interaction_details": {},
        },
        "meta": {"filename": "demo.csv", "n_rows": 10, "n_columns": 3},
        "filename": "report.html",
        "theme": "dark",
    }

    try:
        # success
        out_file = tmp_path / "saved_report.html"

        def _save_ok(name: str, content: bytes):
            out_file.write_bytes(content)
            return str(out_file)

        api_module.configure_desktop_export_hook(_save_ok)
        resp_ok = client.post("/export/html", json=payload)
        assert resp_ok.status_code == 200
        assert resp_ok.json()["saved"] is True
        assert out_file.exists()
        saved = out_file.read_bytes()
        assert saved
        assert b"AdvancedCATDAP Report" in saved
        assert b"<!DOCTYPE html>" in saved

        # cancelled
        api_module.configure_desktop_export_hook(lambda _name, _content: None)
        resp_cancel = client.post("/export/html", json=payload)
        assert resp_cancel.status_code == 200
        body_cancel = resp_cancel.json()
        assert body_cancel["saved"] is False
        assert body_cancel["reason"] == "cancelled"

        # error
        def _save_error(_name: str, _content: bytes):
            raise ValueError("dialog boom")

        api_module.configure_desktop_export_hook(_save_error)
        resp_err = client.post("/export/html", json=payload)
        assert resp_err.status_code == 500
        assert "Export failed" in resp_err.json()["detail"]
    finally:
        api_module.configure_desktop_export_hook(None)
