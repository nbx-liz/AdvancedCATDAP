import pytest
from unittest.mock import MagicMock, patch
import json
import pandas as pd
from advanced_catdap.service.local_worker import run_worker, update_status
from advanced_catdap.service.schema import AnalysisParams

def test_update_status(tmp_path):
    job_file = tmp_path / "job1.json"
    update_status(job_file, "PENDING")
    
    assert job_file.exists()
    data = json.loads(job_file.read_text())
    assert data["status"] == "PENDING"
    
    update_status(job_file, "SUCCESS", result={"ok": True})
    data = json.loads(job_file.read_text())
    assert data["status"] == "SUCCESS"
    assert data["result"] == {"ok": True}

def test_run_worker_full_load(tmp_path):
    # Test path where sample_size is None (full load)
    data_dir = tmp_path
    (data_dir / "jobs").mkdir(exist_ok=True)
    dataset_id = "ds_full"
    
    # Params with no sample_size
    params = AnalysisParams(target_col="t").model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.AnalyzerService") as mock_an_cls:
         
        mock_dm = mock_dm_cls.return_value
        mock_dm.storage_dir = data_dir
        # Mock connection context manager
        mock_con = MagicMock()
        mock_dm._get_connection.return_value = mock_con
        # Mock df return
        mock_con.execute.return_value.df.return_value = pd.DataFrame({"t": [0, 1]})
        
        mock_analyzer = mock_an_cls.return_value
        mock_analyzer.run_analysis.return_value = MagicMock(model_dump=lambda: {"res": 1})
        
        # Create file
        (data_dir / f"{dataset_id}.parquet").touch()
        
        run_worker("job_full", dataset_id, params, str(data_dir))
        
        # Verify con.execute called (proving query path used)
        assert mock_con.execute.called
        assert "SELECT * FROM" in str(mock_con.execute.call_args)

def test_run_worker_success(tmp_path):
    # Setup Data
    data_dir = tmp_path
    (data_dir / "jobs").mkdir()
    
    dataset_id = "ds1"
    # Mock DatasetManager to verify logic without real duckdb here (optional, or integration)
    # We'll use patches to isolate worker logic
    
    params = AnalysisParams(target_col="t").model_dump_json()
    
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm_cls, \
         patch("advanced_catdap.service.local_worker.AnalyzerService") as mock_an_cls:
         
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
        
        # Create dummy parquet for existence check
        (data_dir / f"{dataset_id}.parquet").touch()
        
        run_worker("job1", dataset_id, params, str(data_dir))
        
        # Verify status file was updated to success
        job_file = data_dir / "jobs" / "job1.json"
        assert job_file.exists()
        assert "SUCCESS" in job_file.read_text()

def test_run_worker_failure(tmp_path):
    with patch("advanced_catdap.service.local_worker.DatasetManager") as mock_dm:
        mock_dm.side_effect = Exception("Boom")
        
        valid_params = Settings(target_col="t").model_dump_json() if "Settings" in locals() else '{"target_col": "t"}'
        run_worker("job_fail", "ds", '{"target_col": "t"}', str(tmp_path))
        
def test_update_status_race(tmp_path):
    job_file = tmp_path / "race.json"
    
    # Simulate race where rename fails once then succeeds (mocking replace)
    # Since we can't easily mock pathlib.Path.replace on a specific instance without patching the class or method where it's used
    # We'll just verify the retry logic by mocking the replace method on the Path object created inside the function?
    # Actually update_status uses job_file directly.
    
    # Let's mock Path.replace in local_worker context
    with patch("pathlib.Path.replace") as mock_replace:
        mock_replace.side_effect = [FileExistsError, None] 
        # First call fails, then we check if it tried unlink?
        # The code logic:
        # try: tmp.replace(target) except FileExistsError: if target.exists(): target.unlink(); tmp.rename(target)
        
        # We need actual files to exist for logic to flow or mock exists()
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink") as mock_unlink, \
             patch("pathlib.Path.rename") as mock_rename:
                
            update_status(job_file, "PENDING")
            
            # With retry loop:
            # 1. replace() raises FileExistsError
            # 2. caught, sleep
            # 3. loop retry -> replace() return value (None) -> break
            # So unlink is NOT called.
            
            assert mock_unlink.call_count == 0
            # replace called twice
            assert mock_replace.call_count == 2
            # rename NOT called if replace succeeds on retry
            assert mock_rename.call_count == 0
