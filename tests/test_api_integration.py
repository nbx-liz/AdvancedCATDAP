from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from advanced_catdap.service import api as api_module
from advanced_catdap.service.dataset_manager import DatasetManager
from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import AnalysisParams
from integration_helpers import wait_for_terminal_status


@pytest.mark.integration
def test_api_jobs_integration_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "jobs.db"
    dataset_manager = DatasetManager(storage_dir=tmp_path)
    job_manager = JobManager(db_path=str(db_path))
    monkeypatch.setattr(api_module, "dataset_manager", dataset_manager)
    monkeypatch.setattr(api_module, "job_manager", job_manager)

    client = TestClient(api_module.app)

    # Missing dataset should fail fast at API boundary.
    resp_missing = client.post(
        "/jobs",
        params={"dataset_id": "does_not_exist_ds"},
        json=AnalysisParams(target_col="target").model_dump(),
    )
    assert resp_missing.status_code == 404

    # Success dataset
    ds_ok = "api_integration_ok"
    df_ok = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1] * 8,
            "x1": [1, 2, 3, 4, 5, 6] * 8,
            "x2": ["a", "b", "a", "b", "a", "b"] * 8,
        }
    )
    df_ok.to_parquet(tmp_path / f"{ds_ok}.parquet")

    resp_submit_ok = client.post(
        "/jobs",
        params={"dataset_id": ds_ok},
        json=AnalysisParams(target_col="target").model_dump(),
    )
    assert resp_submit_ok.status_code == 202
    job_ok = resp_submit_ok.json()["job_id"]
    final_ok, seen_ok = wait_for_terminal_status(client, job_ok, timeout_s=25, interval_s=0.5)
    assert final_ok is not None
    assert final_ok["status"] == "SUCCESS"
    assert final_ok.get("result") is not None
    assert seen_ok & {"PENDING", "RUNNING", "PROGRESS"}

    # API-level cache integration: same dataset+params should return same job_id.
    resp_submit_ok_2 = client.post(
        "/jobs",
        params={"dataset_id": ds_ok},
        json=AnalysisParams(target_col="target").model_dump(),
    )
    assert resp_submit_ok_2.status_code == 202
    assert resp_submit_ok_2.json()["job_id"] == job_ok

    # Failure dataset (invalid target causes analyzer failure after submission)
    ds_fail = "api_integration_fail"
    df_fail = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    df_fail.to_parquet(tmp_path / f"{ds_fail}.parquet")

    resp_submit_fail = client.post(
        "/jobs",
        params={"dataset_id": ds_fail},
        json=AnalysisParams(target_col="missing_target").model_dump(),
    )
    assert resp_submit_fail.status_code == 202
    job_fail = resp_submit_fail.json()["job_id"]
    final_fail, _seen_fail = wait_for_terminal_status(client, job_fail, timeout_s=25, interval_s=0.5)
    assert final_fail is not None
    assert final_fail["status"] == "FAILURE"
    assert final_fail.get("error")
