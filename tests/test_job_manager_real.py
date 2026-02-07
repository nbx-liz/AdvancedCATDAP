from unittest.mock import MagicMock, patch
import pytest
import json
import shutil
import sqlite3
from pathlib import Path
from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import AnalysisParams

@pytest.fixture
def clean_job_dir(tmp_path):
    # Use a temp dir for jobs
    db_path = tmp_path / "jobs.db"
    mgr = JobManager(db_path=str(db_path))
    return mgr

def test_job_caching(clean_job_dir):
    mgr = clean_job_dir
    params = AnalysisParams(target_col="target", max_bins=5)
    ds_id = "test_ds_1"
    
    # 1. First Submit
    with patch("subprocess.Popen") as mock_popen:
         # Mock file opening for log
        with patch("builtins.open", create=True) as mock_open:
            job_id_1 = mgr.submit_job(ds_id, params)
            
            assert mock_popen.called
            assert job_id_1 is not None

    # 2. Simulate Job Completion (Success) calling internal DB method
    # 2. Simulate Job Completion (Success) calling internal DB method
    mgr.repository.update_status(job_id_1, "SUCCESS", result=json.dumps({"ok": 1}))
        
    # 3. Second Submit (Same Params)
    with patch("subprocess.Popen") as mock_popen_2:
        job_id_2 = mgr.submit_job(ds_id, params)
        
        # Should return same ID
        assert job_id_2 == job_id_1
        # Should NOT launch subprocess
        assert not mock_popen_2.called

def test_job_different_params(clean_job_dir):
    mgr = clean_job_dir
    ds_id = "test_ds_2"
    
    with patch("subprocess.Popen") as mock_popen:
         with patch("builtins.open", create=True):
            id1 = mgr.submit_job(ds_id, AnalysisParams(target_col="t1"))
            id2 = mgr.submit_job(ds_id, AnalysisParams(target_col="t2"))
            
            assert id1 != id2


def test_cancel_job_not_implemented(clean_job_dir):
    with pytest.raises(NotImplementedError, match="not implemented"):
        clean_job_dir.cancel_job("job-id")
