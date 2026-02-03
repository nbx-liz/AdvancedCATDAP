from celery.result import AsyncResult
from typing import Dict, Any, Optional
import logging

from advanced_catdap.service.worker import celery_app, run_analysis_task
from advanced_catdap.service.schema import AnalysisParams

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages submission and status retrieval of analysis jobs.
    """
    
    def submit_job(self, dataset_id: str, params: AnalysisParams) -> str:
        """
        Submit an analysis job to the worker.
        Returns job_id.
        """
        # Convert params to dict for JSON serialization
        params_dict = params.model_dump()
        
        # Submit task
        task = run_analysis_task.delay(dataset_id, params_dict)
        logger.info(f"Job submitted: {task.id} for dataset {dataset_id}")
        return task.id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current status of a job.
        """
        result = AsyncResult(job_id, app=celery_app)
        
        status = result.status
        meta = result.info # This contains result if READY, or progress info if PROGRESS/FAILURE
        
        response = {
            "job_id": job_id,
            "status": status,
        }
        
        if status == 'PROGRESS':
            response['progress'] = meta
        elif status == 'SUCCESS':
            response['result'] = meta
            # Cleanup result to avoid bloating Redis if needed? Celery handles expiration.
        elif status == 'FAILURE':
             response['error'] = str(meta)
             
        return response

    def cancel_job(self, job_id: str):
        """
        Attempt to cancel a job.
        """
        celery_app.control.revoke(job_id, terminate=True)
