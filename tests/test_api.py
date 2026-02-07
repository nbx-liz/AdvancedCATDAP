from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
import os
import shutil

from advanced_catdap.service import api as api_module
from advanced_catdap.service.api import app
from advanced_catdap.service.schema import AnalysisParams

client = TestClient(app)

@pytest.fixture
def mock_dataset_storage(tmp_path):
    # Mock storage dir to use temp path
    with patch("advanced_catdap.service.api.dataset_manager.storage_dir", tmp_path):
        # We need to make sure dataset_manager in api uses this path
        # But dependencies are instantiated at module level.
        # Ideally we should override dependency, but for MVP singleton pattern, we patch the instance attr
        from advanced_catdap.service.api import dataset_manager
        original_dir = dataset_manager.storage_dir
        dataset_manager.storage_dir = tmp_path
        yield tmp_path
        dataset_manager.storage_dir = original_dir

@pytest.fixture
def mock_job_manager():
    # Mock JobManager internal methods to avoid actual subprocess
    with patch("advanced_catdap.service.api.job_manager") as mock_mgr:
        # Mock submit_job to return a fake ID
        mock_mgr.submit_job.return_value = "test_job_123"
        
        # Mock get_job_status to return SUCCESS
        mock_mgr.get_job_status.return_value = {
            "job_id": "test_job_123",
            "status": "SUCCESS",
            "result": {"feature_importances": []}
        }
        yield mock_mgr

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AdvancedCATDAP API is running"}

def test_upload_and_flow(mock_dataset_storage, mock_job_manager):
    # 1. Upload
    csv_content = "A,B,target\n1,x,0\n2,y,1\n3,x,0"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    
    res_upload = client.post("/datasets", files=files)
    assert res_upload.status_code == 200
    data = res_upload.json()
    assert "dataset_id" in data
    ds_id = data["dataset_id"]
    
    # 2. Get Metadata
    res_meta = client.get(f"/datasets/{ds_id}")
    assert res_meta.status_code == 200
    assert res_meta.json()["n_rows"] == 3
    
    # 3. Preview
    res_prev = client.get(f"/datasets/{ds_id}/preview")
    assert res_prev.status_code == 200
    assert len(res_prev.json()) == 3
    
    # 4. Submit Job
    params = {
        "target_col": "target",
        "max_bins": 5
    }
    res_job = client.post(f"/jobs?dataset_id={ds_id}", json=params)
    assert res_job.status_code == 202
    job_id = res_job.json()["job_id"]
    assert job_id == "test_job_123"
    
    # 5. Check Job Status (Mocked)
    res_status = client.get(f"/jobs/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "SUCCESS"


def test_get_dataset_metadata_is_read_only(mock_dataset_storage, mock_job_manager):
    csv_content = "A,B,target\n1,x,0\n2,y,1"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    upload_res = client.post("/datasets", files=files)
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]

    with patch.object(api_module.dataset_manager, "register_dataset") as mock_register:
        metadata_res = client.get(f"/datasets/{dataset_id}")
        assert metadata_res.status_code == 200
        assert metadata_res.json()["n_rows"] == 2
        mock_register.assert_not_called()
def test_resolve_cors_settings_defaults(monkeypatch):
    monkeypatch.delenv("CATDAP_CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("CATDAP_CORS_ALLOW_CREDENTIALS", raising=False)

    settings = api_module.resolve_cors_settings()
    assert settings["allow_origins"] == [
        "http://127.0.0.1:8050",
        "http://localhost:8050",
    ]
    assert settings["allow_credentials"] is False


def test_resolve_cors_settings_from_env(monkeypatch):
    monkeypatch.setenv(
        "CATDAP_CORS_ALLOW_ORIGINS",
        "https://example.com, https://sub.example.com ",
    )
    monkeypatch.setenv("CATDAP_CORS_ALLOW_CREDENTIALS", "true")

    settings = api_module.resolve_cors_settings()
    assert settings["allow_origins"] == ["https://example.com", "https://sub.example.com"]
    assert settings["allow_credentials"] is True
