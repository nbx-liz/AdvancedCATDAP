from pathlib import Path

import pandas as pd
import pytest

from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.local_worker import run_worker
from advanced_catdap.service.schema import AnalysisParams


def _prepare_pending_job(jm: JobManager, job_id: str, dataset_id: str, params_json: str):
    jm.repository.save_job(job_id, dataset_id, "PENDING", params_json)


@pytest.mark.integration
def test_worker_integration_full_load_and_sample_size(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db_path))

    dataset_id = "worker_integration_ds"
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1] * 10,
            "age": [20, 30, 40, 50, 60, 70] * 10,
            "gender": ["M", "F", "M", "F", "M", "F"] * 10,
        }
    )
    df.to_parquet(tmp_path / f"{dataset_id}.parquet")

    # full load path
    params_full = AnalysisParams(target_col="target").model_dump_json()
    job_full = "worker_full_001"
    _prepare_pending_job(jm, job_full, dataset_id, params_full)
    run_worker(job_full, dataset_id, params_full, str(tmp_path), str(db_path))
    status_full = jm.get_job_status(job_full)
    assert status_full["status"] == "SUCCESS"
    assert status_full.get("result") is not None
    assert status_full.get("progress") is not None

    # sample_size path
    params_sample = AnalysisParams(target_col="target", sample_size=20).model_dump_json()
    job_sample = "worker_sample_001"
    _prepare_pending_job(jm, job_sample, dataset_id, params_sample)
    run_worker(job_sample, dataset_id, params_sample, str(tmp_path), str(db_path))
    status_sample = jm.get_job_status(job_sample)
    assert status_sample["status"] == "SUCCESS"
    assert status_sample.get("result") is not None
    assert status_sample.get("progress") is not None


@pytest.mark.integration
def test_worker_integration_failure_persists_error(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db_path))

    dataset_id = "worker_integration_fail_ds"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    df.to_parquet(tmp_path / f"{dataset_id}.parquet")

    params_bad = AnalysisParams(target_col="missing_target").model_dump_json()
    job_fail = "worker_fail_001"
    _prepare_pending_job(jm, job_fail, dataset_id, params_bad)
    run_worker(job_fail, dataset_id, params_bad, str(tmp_path), str(db_path))

    status = jm.get_job_status(job_fail)
    assert status["status"] == "FAILURE"
    assert status.get("error")
