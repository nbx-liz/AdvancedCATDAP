import json
import uuid
import hashlib
import sys
import logging
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional
from advanced_catdap.service.schema import AnalysisParams
from advanced_catdap.service.executor import JobExecutor, LocalProcessExecutor

from advanced_catdap.service.repository import JobRepository, SQLiteJobRepository

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages job submission via JobExecutor and status tracking via JobRepository.
    """
    def __init__(self, db_path: str = "data/jobs.db", 
                 executor: Optional[JobExecutor] = None,
                 repository: Optional[JobRepository] = None):
        self.db_path = Path(db_path)
        self.data_dir = self.db_path.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor or LocalProcessExecutor()
        self.repository = repository or SQLiteJobRepository(self.db_path)
        self.repository.init_storage()
    
    def submit_job(self, dataset_id: str, params: AnalysisParams) -> str:
        # Create deterministic Job ID based on inputs (Caching)
        # Sort keys to ensure consistent order
        params_json = json.dumps(params.model_dump(), sort_keys=True)
        key_str = f"{dataset_id}|{params_json}"
        job_id = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        # Check if job already exists
        status_info = self.get_job_status(job_id)
        if status_info["status"] in ["PENDING", "RUNNING", "PROGRESS", "SUCCESS"]:
            logger.info(f"Job {job_id} already exists with status {status_info['status']}. Returning cached result.")
            return job_id
        
        # If UNKNOWN (doesn't exist) or FAILURE, we submit a new run
        logger.info(f"Submitting new local job {job_id}")

        # Transaction: Insert PENDING state
        self.repository.save_job(job_id, dataset_id, "PENDING", params_json)
        
        # Prepare execution
        script_path = Path(__file__).parent / "local_worker.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--job-id", job_id,
            "--dataset-id", dataset_id,
            "--params", params_json,
            "--db-path", str(self.db_path),
            "--data-dir", str(self.data_dir)
        ]
        
        log_dir = self.data_dir / "jobs_logs"
        log_file = log_dir / f"{job_id}.log"
        
        # Execute via executor with failure handling
        try:
            self.executor.submit(cmd, log_file)
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            self.repository.update_status(job_id, "FAILURE", error=str(e))
            raise RuntimeError(f"Failed to submit job: {e}") from e
            
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.repository.get_job(job_id)
        if not job:
            return {"job_id": job_id, "status": "UNKNOWN"}
            
        res = dict(job)
        # Deserialize JSON fields
        for field in ['params', 'result', 'progress']:
            if res.get(field):
                try:
                    res[field] = json.loads(res[field])
                except json.JSONDecodeError:
                    pass 
        return res
            
    def cancel_job(self, job_id: str):
        # Local process cancellation is hard without PID tracking.
        # For MVP, we just ignore it.
        pass
