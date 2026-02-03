import os
import logging
from celery import Celery
from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.dataset_manager import DatasetManager
from advanced_catdap.service.schema import AnalysisParams, AnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis URL (fallback to localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery
celery_app = Celery(
    "advanced_catdap_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Windows support optimizations (if needed)
)

@celery_app.task(bind=True, name="run_analysis_task")
def run_analysis_task(self, dataset_id: str, params_dict: dict):
    """
    Celery task to run AdvancedCATDAP analysis.
    """
    logger.info(f"Task started: dataset={dataset_id}")
    
    try:
        # Rehydrate params
        params = AnalysisParams(**params_dict)
        
        # Initialize Managers
        # TODO: Configure storage dir from env or config
        dataset_manager = DatasetManager(storage_dir="data")
        analyzer = AnalyzerService()
        
        # Get data
        # For analysis, we might want the full dataset or a large sample
        # If dataset is huge, 'get_sample' using stratify or full load might be better
        # Here we load full dataset via pandas for now (Core requirement)
        # Warning: Memory intensive! 
        # Future optimization: Partial read in Core or iterate.
        # Check if sampling is requested in params
        if params.sample_size:
            logger.info(f"Loading sample of {params.sample_size} rows")
            df = dataset_manager.get_sample(dataset_id, n_rows=params.sample_size)
        else:
            logger.info("Loading full dataset")
            # We can use get_sample with a very large number/all
            # Or implement get_full in manager. 
            # Reusing get_sample for now with default large limit or implementing get_all
            df = dataset_manager.con.execute(f"SELECT * FROM '{dataset_manager.storage_dir}/{dataset_id}.parquet'").df()

        # Define progress callback to update Celery state
        def progress_tracker(stage, data):
            # Update task state
            # meta can be retrieved by client
            self.update_state(state='PROGRESS', meta={
                'stage': stage,
                'data': data
            })
            logger.info(f"Progress: {stage} - {data}")

        # Run Analysis
        result = analyzer.run_analysis(df, params, progress_cb=progress_tracker)
        
        logger.info("Analysis completed successfully")
        return result.model_dump()

    except Exception as e:
        logger.error(f"Analysis task failed: {e}")
        # Re-raise so Celery marks it as FAILED
        raise e
