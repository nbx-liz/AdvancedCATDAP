from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.local_worker import run_worker
from advanced_catdap.service.schema import AnalysisParams
from integration_helpers import wait_for_terminal_status


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


@pytest.mark.integration
def test_windows_main_worker_mode_runs_job(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db_path))

    dataset_id = "worker_entry_ds"
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "x": [10, 20, 30, 40, 50, 60],
        }
    )
    df.to_parquet(tmp_path / f"{dataset_id}.parquet")

    params_json = AnalysisParams(target_col="target").model_dump_json()
    job_id = "worker_entry_001"
    _prepare_pending_job(jm, job_id, dataset_id, params_json)

    cmd = [
        sys.executable,
        "scripts/windows_main.py",
        "--worker",
        "--job-id",
        job_id,
        "--dataset-id",
        dataset_id,
        "--params",
        params_json,
        "--data-dir",
        str(tmp_path),
        "--db-path",
        str(db_path),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    status = jm.get_job_status(job_id)
    assert status["status"] in {"SUCCESS", "FAILURE"}
    assert status["status"] != "PENDING"


@pytest.mark.integration
def test_worker_failure_via_job_manager_persists_error_and_log(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db_path))

    dataset_id = "worker_failure_log_ds"
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    df.to_parquet(tmp_path / f"{dataset_id}.parquet")

    # Submit with missing target to force analyzer failure in worker process.
    job_id = jm.submit_job(dataset_id, AnalysisParams(target_col="missing_target"))
    final_status, _seen = wait_for_terminal_status(jm, job_id, timeout_s=30, interval_s=0.2)

    assert final_status is not None
    assert final_status["status"] == "FAILURE"
    assert final_status.get("error")
    assert final_status.get("result") in (None, "")

    log_file = tmp_path / "jobs_logs" / f"{job_id}.log"
    assert log_file.exists()
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    assert "missing_target" in text or "Target column" in text
