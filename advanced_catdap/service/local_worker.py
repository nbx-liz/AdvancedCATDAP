import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.dataset_manager import DatasetManager
from advanced_catdap.service.schema import AnalysisParams

def update_status(job_file: Path, status: str, result=None, progress=None, error=None):
    """
    Atomic update of job file.
    """
    data = {
        "job_id": job_file.stem,
        "status": status,
        "last_updated": datetime.now().isoformat()
    }
    if result: data["result"] = result
    if progress: data["progress"] = progress
    if error: data["error"] = error
    
    # Write to temp then move to ensure atomic read
    tmp_path = job_file.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, default=str)
    
    # Simple rename (atomic on POSIX, usually ok on Windows for replace)
    try:
        tmp_path.replace(job_file)
    except FileExistsError:
        # On Windows, replace might raise if file exists and is locked.
        # Retry or remove first
        if job_file.exists():
            job_file.unlink()
        tmp_path.rename(job_file)

def run_worker(job_id: str, dataset_id: str, params_json: str, data_dir: str):
    job_file = Path(data_dir) / "jobs" / f"{job_id}.json"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting job {job_id} for dataset {dataset_id}")
    
    try:
        # Initial running state
        update_status(job_file, "RUNNING")

        # Parse params
        params_dict = json.loads(params_json)
        params = AnalysisParams(**params_dict)

        # Managers
        dataset_manager = DatasetManager(storage_dir=Path(data_dir))
        analyzer = AnalyzerService()

        # Load Data
        # Using get_sample logic (all data if no sample size)
        if params.sample_size:
            df = dataset_manager.get_sample(dataset_id, n_rows=params.sample_size)
        else:
            # Full load via duckdb query
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
            print(f"Progress: {stage} - {data}")
            update_status(job_file, "PROGRESS", progress={"stage": stage, "data": data})

        # Run
        result = analyzer.run_analysis(df, params, progress_cb=progress_tracker)
        
        from advanced_catdap.service.utils import sanitize_for_json
        clean_result = sanitize_for_json(result.model_dump())

        # Success
        update_status(job_file, "SUCCESS", result=clean_result)
        print("Job successful")

    except Exception as e:
        err_msg = str(e)
        st = traceback.format_exc()
        print(f"Job Failed: {err_msg}\n{st}")
        update_status(job_file, "FAILURE", error=err_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--data-dir", default="data")
    
    args = parser.parse_args()
    
    run_worker(
        job_id=args.job_id,
        dataset_id=args.dataset_id,
        params_json=args.params,
        data_dir=args.data_dir
    )
