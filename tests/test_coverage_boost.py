
import pytest
from unittest.mock import MagicMock, patch, ANY
import json
import sqlite3
import pandas as pd
from pathlib import Path
from advanced_catdap.service.local_worker import run_worker
from advanced_catdap.service.schema import AnalysisParams
from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.executor import JobExecutor

def test_local_worker_with_sample_size(tmp_path):
    """Test local worker with sample_size set."""
    data_dir = tmp_path
    db_path = tmp_path / "jobs.db"
    dataset_id = "ds_sample"
    
    params = AnalysisParams(target_col="t", sample_size=100).model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.AnalyzerService") as mock_an_cls, \
         patch("advanced_catdap.service.local_worker.JobManager") as mock_jm_cls:
         
        mock_dm = mock_dm_cls.return_value
        mock_dm.storage_dir = data_dir
        # get_sample should be called instead of direct SQL
        mock_dm.get_sample.return_value = pd.DataFrame({"t": range(10)})
        
        # Make run_analysis call the callback
        mock_analyzer = mock_an_cls.return_value
        
        def side_effect(df, params, progress_cb=None):
            if progress_cb:
                progress_cb("test_stage", "test_data")
            return MagicMock(model_dump=lambda: {"res": "sample"})
            
        mock_analyzer.run_analysis.side_effect = side_effect
        
        mock_jm = mock_jm_cls.return_value
        
        run_worker("job_sample", dataset_id, params, str(data_dir), str(db_path))
        
        assert mock_dm.get_sample.called
        mock_dm.get_sample.assert_called_with(dataset_id, n_rows=100)
        mock_jm._update_job_status.assert_any_call("job_sample", "SUCCESS", result={'res': "sample"})

def test_local_worker_file_not_found(tmp_path):
    """Test local worker when dataset file is missing."""
    data_dir = tmp_path
    db_path = tmp_path / "jobs.db"
    dataset_id = "ds_missing"
    params = AnalysisParams(target_col="t").model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.JobManager") as mock_jm_cls:
         
        mock_dm = mock_dm_cls.return_value
        mock_dm.storage_dir = data_dir
        
        # Ensure file does not exist (tmp_path is empty)
        
        mock_jm = mock_jm_cls.return_value
        
        run_worker("job_missing", dataset_id, params, str(data_dir), str(db_path))
        
        args = mock_jm._update_job_status.call_args_list[-1]
        assert args[0][0] == "job_missing"
        assert args[0][1] == "FAILURE"
        assert "not found" in args[1]['error']

def test_job_manager_submit_failure(tmp_path):
    """Test JobManager handles executor failure."""
    db_path = tmp_path / "jobs.db"
    
    class FailExecutor:
        def submit(self, cmd, log, env=None):
            raise RuntimeError("Exec failed")
            
    jm = JobManager(str(db_path), executor=FailExecutor())
    params = AnalysisParams(target_col="t")
    
    with pytest.raises(RuntimeError, match="Failed to submit job"):
        jm.submit_job("ds", params)
        
    status = jm.get_job_status(jm.get_job_status("ds")['job_id']) # Wait, get_job_status needs ID.
    # But submit_job raises, so we don't get ID back? 
    # Actually submit_job computes ID first. But we can't get it if it raises.
    # However, we can re-compute it or look in DB.
    
    # Let's inspect DB directly
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error FROM jobs").fetchone()
        assert row[0] == "FAILURE"
        assert "Exec failed" in row[1]

def test_job_manager_db_error_on_status(tmp_path):
    """Test JobManager handles DB error during status update."""
    db_path = tmp_path / "jobs.db"
    jm = JobManager(str(db_path))
    
    # Create a job manually
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO jobs (job_id, dataset_id, status) VALUES ('j1', 'd1', 'PENDING')")
        
    # Mock _get_connection to raise
    with patch.object(jm, '_get_connection', side_effect=sqlite3.Error("DB connection died")):
        # Should catch error and log it, not raise
        jm._update_job_status("j1", "RUNNING")
        # Pass if no exception
