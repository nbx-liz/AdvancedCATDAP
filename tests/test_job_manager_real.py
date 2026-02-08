from unittest.mock import MagicMock, patch
import pytest
import json
import shutil
import sqlite3
import sys
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


def test_submit_job_frozen_mode_builds_worker_command(tmp_path, monkeypatch):
    captured = {}

    class _CaptureExecutor:
        def submit(self, cmd, log_file):
            captured["cmd"] = cmd
            captured["log_file"] = str(log_file)

    db_path = tmp_path / "jobs.db"
    mgr = JobManager(db_path=str(db_path), executor=_CaptureExecutor())
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    job_id = mgr.submit_job("dataset_frozen", AnalysisParams(target_col="target"))

    cmd = captured["cmd"]
    assert job_id
    assert "--worker" in cmd
    assert cmd[0] == sys.executable
    assert "--job-id" in cmd
    assert "--db-path" in cmd


def test_get_job_status_keeps_invalid_json_fields_as_raw_string(tmp_path):
    db_path = tmp_path / "jobs.db"
    mgr = JobManager(db_path=str(db_path))
    mgr.repository.save_job("j_invalid", "d", "SUCCESS", "{}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "UPDATE jobs SET params=?, result=?, progress=? WHERE job_id=?",
        ('{"ok":1}', "{invalid-json", '{"stage":bad}', "j_invalid"),
    )
    con.commit()
    con.close()

    status = mgr.get_job_status("j_invalid")
    assert isinstance(status["params"], dict)
    assert status["result"] == "{invalid-json"
    assert status["progress"] == '{"stage":bad}'
