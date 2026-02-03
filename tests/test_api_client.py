import pytest
import httpx
from unittest.mock import MagicMock, patch
from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import DatasetMetadata, AnalysisParams

@pytest.fixture
def client():
    return APIClient(base_url="http://test")

def test_upload_dataset(client):
    with patch("httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dataset_id": "123", "filename": "t.csv", "file_path": "p", 
            "n_rows": 10, "n_columns": 2, "columns": []
        }
        mock_post.return_value = mock_resp
        
        file_obj = MagicMock()
        res = client.upload_dataset(file_obj, "t.csv")
        assert isinstance(res, DatasetMetadata)
        assert res.dataset_id == "123"

def test_upload_error(client):
    with patch("httpx.post") as mock_post:
        mock_post.side_effect = httpx.HTTPError("Boom")
        
        with pytest.raises(RuntimeError):
            client.upload_dataset(MagicMock(), "t.csv")

def test_get_preview(client):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [{"a": 1}]
        
        res = client.get_preview("123")
        assert len(res) == 1

def test_submit_job(client):
    with patch("httpx.post") as mock_post:
        mock_post.return_value.status_code = 202
        mock_post.return_value.json.return_value = {"job_id": "j1"}
        
        res = client.submit_job("d1", AnalysisParams(target_col="t"))
        assert res == "j1"

def test_get_dataset_success(client):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "dataset_id": "d1", "filename": "f", "file_path": "p", "n_rows": 1, "n_columns": 1, "columns": []
        }
        res = client.get_dataset("d1")
        assert res.dataset_id == "d1"

def test_get_dataset_error(client):
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.HTTPError("Boom")
        with pytest.raises(RuntimeError):
            client.get_dataset("d1")

def test_get_preview_error(client):
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.HTTPError("Boom")
        with pytest.raises(RuntimeError):
            client.get_preview("d1")

def test_submit_job_error(client):
    with patch("httpx.post") as mock_post:
        mock_post.side_effect = httpx.HTTPError("Boom")
        with pytest.raises(RuntimeError):
            client.submit_job("d1", AnalysisParams(target_col="t"))

def test_get_job_status_error(client):
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.HTTPError("Boom")
        with pytest.raises(RuntimeError):
            client.get_job_status("j1")

def test_get_job_status_success(client):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "SUCCESS"}
        res = client.get_job_status("j1")
        assert res["status"] == "SUCCESS"
