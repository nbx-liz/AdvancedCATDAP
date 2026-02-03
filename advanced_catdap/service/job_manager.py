import subprocess
import json
import uuid
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
        job_id = str(uuid.uuid4())
        
        # We invoke the local_worker.py script
        # Using sys.executable to ensure we use the same python env
        script_path = Path(__file__).parent / "local_worker.py"
        
        params_json = json.dumps(params.model_dump())
        
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
            
        logger.info(f"Submitted local job {job_id}")
        
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
