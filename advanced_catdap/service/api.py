from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any
import shutil
import os
import tempfile
from pathlib import Path
import pandas as pd
import logging

from advanced_catdap.service.dataset_manager import DatasetManager
from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import DatasetMetadata, AnalysisParams, AnalysisResult

app = FastAPI(title="AdvancedCATDAP API", version="0.1.0")
logger = logging.getLogger(__name__)

DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://127.0.0.1:8050",
    "http://localhost:8050",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_cors_settings() -> dict[str, Any]:
    raw_origins = os.environ.get("CATDAP_CORS_ALLOW_ORIGINS", "")
    if raw_origins.strip():
        allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        allow_origins = list(DEFAULT_CORS_ALLOW_ORIGINS)

    allow_credentials = _env_bool("CATDAP_CORS_ALLOW_CREDENTIALS", False)
    if allow_origins == ["*"] and allow_credentials:
        logger.warning("allow_credentials=True with wildcard origin is insecure.")

    return {
        "allow_origins": allow_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }

# CORS setup (allow all for local dev)
cors_settings = resolve_cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_settings["allow_origins"],
    allow_credentials=cors_settings["allow_credentials"],
    allow_methods=cors_settings["allow_methods"],
    allow_headers=cors_settings["allow_headers"],
)

# Dependencies (Simple Singleton pattern for MVP)
dataset_manager = DatasetManager(storage_dir="data")
job_manager = JobManager()

@app.get("/")
def read_root():
    return {"message": "AdvancedCATDAP API is running"}

# --- Datasets ---

@app.post("/datasets", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV/Parquet file and register it.
    """
    suffix = Path(file.filename).suffix
    if suffix.lower() not in ['.csv', '.parquet']:
        raise HTTPException(status_code=400, detail="Only .csv and .parquet files are supported.")
    
    # Save to temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        metadata = dataset_manager.register_dataset(tmp_path, original_filename=file.filename)
        return metadata
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/datasets/{dataset_id}", response_model=DatasetMetadata)
def get_dataset_metadata(dataset_id: str):
    path = dataset_manager.storage_dir / f"{dataset_id}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # In a production app, we would query a database.
    # Here we re-scan the file to reconstruct metadata.
    return dataset_manager.register_dataset(str(path), dataset_id=dataset_id) 


@app.get("/datasets/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, rows: int = 100):
    try:
        df = dataset_manager.get_preview(dataset_id, n_rows=rows)
        # Handle NaN -> None for valid JSON (Robust)
        df = df.astype(object).where(pd.notnull(df), None)
        # Convert to dict for JSON
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@app.get("/datasets/{dataset_id}/sample")
def get_dataset_sample(dataset_id: str, rows: int = 1000):
    try:
        df = dataset_manager.get_sample(dataset_id, n_rows=rows)
        df = df.astype(object).where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")


# --- Jobs ---

@app.post("/jobs", status_code=202)
def submit_job(dataset_id: str, params: AnalysisParams):
    path = dataset_manager.storage_dir / f"{dataset_id}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Ensure target exists in dataset? 
    # AnalyzerService will fail if not, which is fine (Job FAILED).
    
    job_id = job_manager.submit_job(dataset_id, params)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    return job_manager.get_job_status(job_id)

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    try:
        job_manager.cancel_job(job_id)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return {"status": "cancelled"}
