from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from advanced_catdap.service import api as api_module


@pytest.mark.integration
def test_desktop_export_hook_success_cancel_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        saved_file = tmp_path / "saved_report.html"

        def _ok(_name: str, content: bytes):
            saved_file.write_bytes(content)
            return str(saved_file)

        api_module.configure_desktop_export_hook(_ok)
        resp_ok = client.post("/export/html", json=payload)
        assert resp_ok.status_code == 200
        body_ok = resp_ok.json()
        assert body_ok["saved"] is True
        assert saved_file.exists()

        api_module.configure_desktop_export_hook(lambda _name, _content: None)
        resp_cancel = client.post("/export/html", json=payload)
        assert resp_cancel.status_code == 200
        body_cancel = resp_cancel.json()
        assert body_cancel["saved"] is False
        assert body_cancel["reason"] == "cancelled"

        def _boom(_name: str, _content: bytes):
            raise RuntimeError("boom")

        api_module.configure_desktop_export_hook(_boom)
        resp_err = client.post("/export/html", json=payload)
        assert resp_err.status_code == 500
        assert "Export failed" in resp_err.json()["detail"]
    finally:
        api_module.configure_desktop_export_hook(None)

