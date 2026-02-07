import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.dataset_manager import DatasetManager
from advanced_catdap.service.schema import AnalysisParams
from advanced_catdap.service.job_manager import JobManager

logger = logging.getLogger(__name__)

def run_worker(job_id: str, dataset_id: str, params_json: str, data_dir: str, db_path: str):
    
    logger.info("Starting job %s for dataset %s", job_id, dataset_id)
    
    # Initialize JobManager
    job_manager = JobManager(db_path=db_path)
    
    try:
        # Initial running state
        job_manager.repository.update_status(job_id, "RUNNING")

        # Parse params
        params_dict = json.loads(params_json)
        params = AnalysisParams(**params_dict)

        # Managers
        dataset_manager = DatasetManager(storage_dir=Path(data_dir))
        analyzer = AnalyzerService()

        # Load Data
        if params.sample_size:
            df = dataset_manager.get_sample(dataset_id, n_rows=params.sample_size)
        else:
            path = dataset_manager.storage_dir / f"{dataset_id}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Dataset {dataset_id} not found")
            
            con = dataset_manager._get_connection()
            try:
                df = con.execute(f"SELECT * FROM '{path}'").df()
            finally:
                con.close()

        # Callback
        def progress_tracker(stage, data):
            logger.debug("Progress: %s - %s", stage, data)
            from advanced_catdap.service.utils import sanitize_for_json
            clean_data = sanitize_for_json(data)
            job_manager.repository.update_status(job_id, "PROGRESS", progress=json.dumps({"stage": stage, "data": clean_data}))

        # Run
        result = analyzer.run_analysis(df, params, progress_cb=progress_tracker)
        
        from advanced_catdap.service.utils import sanitize_for_json
        clean_result = sanitize_for_json(result.model_dump())

        # Success
        job_manager.repository.update_status(job_id, "SUCCESS", result=json.dumps(clean_result))
        logger.info("Job successful")

    except Exception as e:
        err_msg = str(e)
        logger.exception("Job failed: %s", err_msg)
        job_manager.repository.update_status(job_id, "FAILURE", error=err_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db-path", default="data/jobs.db")
    
    args = parser.parse_args()
    
    run_worker(
        job_id=args.job_id,
        dataset_id=args.dataset_id,
        params_json=args.params,
        data_dir=args.data_dir,
        db_path=args.db_path
    )
