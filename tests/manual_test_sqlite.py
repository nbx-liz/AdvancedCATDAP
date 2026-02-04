import sys
import time
import json
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import AnalysisParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sqlite_jobs():
    # Setup - use a test db in a temp-like location to avoid messing with real data
    # ensuring we use absolute paths might differ, but let's stick to relative 'data' for consistency with defaults if possible
    # specific db for testing
    db_path = Path("data/test_jobs.db")
    data_dir = db_path.parent
    data_dir.mkdir(exist_ok=True)
    
    # Clean up previous db
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            print("Could not delete old db, ignoring")
        
    print(f"Initializing JobManager with {db_path}...")
    jm = JobManager(db_path=str(db_path))
    
    # Create a dummy dataset
    dataset_id = "manual_test_ds"
    parquet_path = data_dir / f"{dataset_id}.parquet"
    
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1]*10,
        "feature1": [1, 2, 3, 4, 5, 6]*10,
        "feature2": ["a", "b", "a", "b", "a", "b"]*10
    })
    df.to_parquet(parquet_path)
    print(f"Created dummy dataset at {parquet_path}")

    params = AnalysisParams(target_col="target")
    
    print("Submitting job...")
    job_id = jm.submit_job(dataset_id, params)
    print(f"Job submitted: {job_id}")
    
    # Test 2: Check status immediately
    status = jm.get_job_status(job_id)
    print(f"Immediate status: {status}")
    assert status['status'] == 'PENDING'
    
    # Wait for worker
    print("Waiting for worker...")
    final_status = None
    for i in range(20):
        time.sleep(1)
        status = jm.get_job_status(job_id)
        print(f"[{i}s] Status: {status['status']}")
        if status['status'] in ['SUCCESS', 'FAILURE']:
            final_status = status
            break
            
    if final_status is None:
        print("Timeout waiting for job completion")
        # Print logs
        log_file = data_dir / "jobs_logs" / f"{job_id}.log"
        if log_file.exists():
            print("Worker Log:")
            print(log_file.read_text())
        else:
            print("No log file found.")
        sys.exit(1)
        
    print(f"Final Status: {final_status}")
    
    if final_status['status'] == 'FAILURE':
        print(f"Job failed: {final_status.get('error')}")
        # Print logs
        log_file = data_dir / "jobs_logs" / f"{job_id}.log"
        if log_file.exists():
            print("Worker Log:")
            print(log_file.read_text())
        sys.exit(1)
    
    assert final_status['status'] == 'SUCCESS'
    print("Job succeeded!")
    print("Result snippet:", str(final_status.get('result'))[:200])
    
    # Cleanup
    if parquet_path.exists():
        parquet_path.unlink()

    print("SQLite verification passed!")

if __name__ == "__main__":
    test_sqlite_jobs()
