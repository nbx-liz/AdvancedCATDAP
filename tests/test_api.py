from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
import os
import shutil

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
def mock_celery():
    with patch("advanced_catdap.service.job_manager.run_analysis_task") as mock_task:
        mock_res = MagicMock()
        mock_res.id = "test_job_123"
        mock_task.delay.return_value = mock_res
        yield mock_task

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AdvancedCATDAP API is running"}

def test_upload_and_flow(mock_dataset_storage, mock_celery):
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
    
    # 5. Check Job Status (mocked via JobManager logic which queries Celery)
    # We need to mock AsyncResult in JobManager.get_job_status
    with patch("advanced_catdap.service.job_manager.AsyncResult") as mock_res_cls:
        mock_instance = MagicMock()
        mock_instance.status = "SUCCESS"
        mock_instance.info = {"some": "result"}
        mock_res_cls.return_value = mock_instance
        
        res_status = client.get(f"/jobs/{job_id}")
        assert res_status.status_code == 200
        assert res_status.json()["status"] == "SUCCESS"
