import subprocess
import json
import uuid
import hashlib
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from advanced_catdap.service.schema import AnalysisParams

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages job submission via local subprocess and file-based status tracking.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
    
    def submit_job(self, dataset_id: str, params: AnalysisParams) -> str:
        # Create deterministic Job ID based on inputs (Caching)
        # Sort keys to ensure consistent order
        params_json = json.dumps(params.model_dump(), sort_keys=True)
        key_str = f"{dataset_id}|{params_json}"
        job_id = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        # Check if job already exists
        existing_status = self.get_job_status(job_id)
        if existing_status["status"] in ["PENDING", "RUNNING", "PROGRESS", "SUCCESS"]:
            logger.info(f"Job {job_id} already exists with status {existing_status['status']}. Returning cached result.")
            return job_id
        
        # If UNKNOWN (doesn't exist) or FAILURE, we submit a new run
        logger.info(f"Submitting new local job {job_id} (Status: {existing_status['status']})")
        
        # We invoke the local_worker.py script
        # Using sys.executable to ensure we use the same python env
        script_path = Path(__file__).parent / "local_worker.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--job-id", job_id,
            "--dataset-id", dataset_id,
            "--params", params_json,
            "--data-dir", str(self.data_dir)
        ]
        
        # Popen is non-blocking
        # We redirect stdout/stderr to a log file for debugging
        log_file = self.jobs_dir / f"{job_id}.log"
        with open(log_file, "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=f)
            
        # Create initial PENDING status file immediately so API doesn't 404
        self._write_initial_status(job_id)
        
        return job_id

    def _write_initial_status(self, job_id: str):
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            with open(job_file, "w") as f:
                json.dump({"job_id": job_id, "status": "PENDING"}, f)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not job_file.exists():
            # Check if log exists, maybe it crashed before writing json?
            log_file = self.jobs_dir / f"{job_id}.log"
            if log_file.exists():
                 return {"job_id": job_id, "status": "PENDING", "note": "Processing..."}
            return {"job_id": job_id, "status": "UNKNOWN"}
            
        try:
            with open(job_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"job_id": job_id, "status": "PENDING", "note": "Reading..."}
            
    def cancel_job(self, job_id: str):
        # Local process cancellation is hard without PID tracking.
        # For MVP, we just ignore it.
        pass
