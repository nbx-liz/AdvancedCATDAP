import subprocess
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

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages job submission via local subprocess and SQLite status tracking.
    """
    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = Path(db_path)
        self.data_dir = self.db_path.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with self._get_connection() as conn:
            # Automatic transaction via 'with conn' inside _get_connection not guaranteed if we change it
            # Explicitly commit here or rely on the helper
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params TEXT,
                    result TEXT,
                    error TEXT,
                    progress TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Yields a connection that is automatically closed on exit."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()
    
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

        with self._get_connection() as conn:
            with conn: # Transaction
                conn.execute("""
                    INSERT OR REPLACE INTO jobs (job_id, dataset_id, status, params, updated_at) 
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (job_id, dataset_id, "PENDING", params_json))
        
        # We invoke the local_worker.py script
        # Using sys.executable to ensure we use the same python env
        script_path = Path(__file__).parent / "local_worker.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--job-id", job_id,
            "--dataset-id", dataset_id,
            "--params", params_json,
            "--db-path", str(self.db_path),
            # Pass data dir for DatasetManager
            "--data-dir", str(self.data_dir)
        ]
        
        # Popen is non-blocking
        # We redirect stdout/stderr to a log file for debugging
        log_dir = self.data_dir / "jobs_logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{job_id}.log"
        
        with open(log_file, "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=f)
            
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if not row:
                return {"job_id": job_id, "status": "UNKNOWN"}
            
            res = dict(row)
            # Deserialize JSON fields
            for field in ['params', 'result', 'progress']:
                if res.get(field):
                    try:
                        res[field] = json.loads(res[field])
                    except json.JSONDecodeError:
                        pass 
            
            return res
            
    def _update_job_status(self, job_id: str, status: str, result=None, error=None, progress=None):
        query_parts = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if result is not None:
            query_parts.append("result = ?")
            params.append(json.dumps(result))
        
        if error is not None:
            query_parts.append("error = ?")
            params.append(error)
            
        if progress is not None:
            query_parts.append("progress = ?")
            params.append(json.dumps(progress))
            
        params.append(job_id)
        
        sql = f"UPDATE jobs SET {', '.join(query_parts)} WHERE job_id = ?"
        
        with self._get_connection() as conn:
            with conn: # Transaction
                conn.execute(sql, params)

    def cancel_job(self, job_id: str):
        # Local process cancellation is hard without PID tracking.
        # For MVP, we just ignore it.
        pass
