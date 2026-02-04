import pytest
from unittest.mock import MagicMock, patch
import json
import pandas as pd
from advanced_catdap.service.local_worker import run_worker
from advanced_catdap.service.schema import AnalysisParams

def test_run_worker_full_load(tmp_path):
    # Test path where sample_size is None (full load)
    data_dir = tmp_path
    db_path = tmp_path / "jobs.db"
    (data_dir / "jobs").mkdir(exist_ok=True)
    dataset_id = "ds_full"
    
    # Params with no sample_size
    params = AnalysisParams(target_col="t").model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.AnalyzerService") as mock_an_cls, \
         patch("advanced_catdap.service.local_worker.JobManager") as mock_jm_cls:
         
        mock_dm = mock_dm_cls.return_value
        mock_dm.storage_dir = data_dir
        # Mock connection context manager
        mock_con = MagicMock()
        mock_dm._get_connection.return_value = mock_con
        # Mock df return
        mock_con.execute.return_value.df.return_value = pd.DataFrame({"t": [0, 1]})
        
        mock_analyzer = mock_an_cls.return_value
        mock_analyzer.run_analysis.return_value = MagicMock(model_dump=lambda: {"res": 1})

        mock_jm = mock_jm_cls.return_value
        
        # Create file
        (data_dir / f"{dataset_id}.parquet").touch()
        
        run_worker("job_full", dataset_id, params, str(data_dir), str(db_path))
        
        # Verify con.execute called (proving query path used)
        assert mock_con.execute.called
        assert "SELECT * FROM" in str(mock_con.execute.call_args)
        
        # Verify status updates
        assert mock_jm._update_job_status.call_count >= 2 # RUNNING, SUCCESS (maybe progress)
        mock_jm._update_job_status.assert_any_call("job_full", "RUNNING")
        mock_jm._update_job_status.assert_any_call("job_full", "SUCCESS", result={'res': 1})

def test_run_worker_success(tmp_path):
    # Setup Data
    data_dir = tmp_path
    db_path = tmp_path / "jobs.db"
    (data_dir / "jobs").mkdir(exist_ok=True)
    
    dataset_id = "ds1"
    
    params = AnalysisParams(target_col="t").model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.AnalyzerService") as mock_an_cls, \
         patch("advanced_catdap.service.local_worker.JobManager") as mock_jm_cls:
         
        mock_dm = mock_dm_cls.return_value
        mock_dm.storage_dir = data_dir
        # Mock connection and DF for full load path
        mock_con = MagicMock()
        mock_dm._get_connection.return_value = mock_con
        mock_con.execute.return_value.df.return_value = pd.DataFrame({"t": [0, 1]})
        
        mock_analyzer = mock_an_cls.return_value
        mock_res = MagicMock()
        mock_res.model_dump.return_value = {"feature_importances": []}
        mock_analyzer.run_analysis.return_value = mock_res
        
        mock_jm = mock_jm_cls.return_value
        
        # Create dummy parquet for existence check
        (data_dir / f"{dataset_id}.parquet").touch()
        
        run_worker("job1", dataset_id, params, str(data_dir), str(db_path))
        
        # Verify success update
        mock_jm._update_job_status.assert_any_call("job1", "SUCCESS", result={'feature_importances': []})

def test_run_worker_failure(tmp_path):
    data_dir = tmp_path
    db_path = tmp_path / "jobs.db"
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm, \
         patch("advanced_catdap.service.local_worker.JobManager") as mock_jm_cls:
        
        mock_dm.side_effect = Exception("Boom")
        mock_jm = mock_jm_cls.return_value
        
        run_worker("job_fail", "ds", '{"target_col": "t"}', str(data_dir), str(db_path))
        
        # Verify failure update
        # We need to check if assert_any_call works with partial matches or check args manually
        # mock_jm._update_job_status.assert_any_call("job_fail", "FAILURE", error="Boom") # exact error string might vary
        
        args = mock_jm._update_job_status.call_args_list[-1]
        assert args[0][0] == "job_fail"
        assert args[0][1] == "FAILURE"
        assert "Boom" in args[1]['error']
