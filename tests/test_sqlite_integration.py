from pathlib import Path

import pandas as pd
import pytest

from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import AnalysisParams
from tests.integration_helpers import wait_for_terminal_status


@pytest.mark.integration
def test_sqlite_jobs_integration(tmp_path: Path) -> None:
    """Integration test for local sqlite-backed job lifecycle."""
    db_path = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db_path))

    dataset_id = "integration_test_ds"
    parquet_path = tmp_path / f"{dataset_id}.parquet"
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1] * 10,
            "feature1": [1, 2, 3, 4, 5, 6] * 10,
            "feature2": ["a", "b", "a", "b", "a", "b"] * 10,
        }
    )
    df.to_parquet(parquet_path)

    params = AnalysisParams(target_col="target")
    job_id = jm.submit_job(dataset_id, params)
    immediate = jm.get_job_status(job_id)
    assert immediate["status"] in {"PENDING", "RUNNING", "PROGRESS"}

    final_status, seen = wait_for_terminal_status(jm, job_id, timeout_s=20, interval_s=0.5)

    if final_status is None:
        log_file = tmp_path / "jobs_logs" / f"{job_id}.log"
        detail = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "no worker log"
        pytest.fail(f"Timeout waiting for job completion. job_id={job_id}\n{detail}")

    if final_status["status"] == "FAILURE":
        log_file = tmp_path / "jobs_logs" / f"{job_id}.log"
        detail = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "no worker log"
        pytest.fail(f"Job failed: {final_status.get('error')}\n{detail}")

    assert final_status["status"] == "SUCCESS"
    assert final_status.get("result") is not None
    assert seen & {"PROGRESS", "RUNNING", "PENDING"}

    # Cache integration: same dataset+params should resolve to same job ID.
    cached_job_id = jm.submit_job(dataset_id, params)
    assert cached_job_id == job_id
