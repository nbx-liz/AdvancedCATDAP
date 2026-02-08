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
def test_jobs_api_e2e_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "jobs.db"
    dataset_manager = DatasetManager(storage_dir=tmp_path)
    job_manager = JobManager(db_path=str(db_path))
    monkeypatch.setattr(api_module, "dataset_manager", dataset_manager)
    monkeypatch.setattr(api_module, "job_manager", job_manager)
    client = TestClient(api_module.app)

    ds_ok = "jobs_api_ok"
    pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1] * 8,
            "x1": [1, 2, 3, 4, 5, 6] * 8,
            "x2": ["a", "b", "a", "b", "a", "b"] * 8,
        }
    ).to_parquet(tmp_path / f"{ds_ok}.parquet")

    resp_ok = client.post(
        "/jobs",
        params={"dataset_id": ds_ok},
        json=AnalysisParams(target_col="target").model_dump(),
    )
    assert resp_ok.status_code == 202
    job_ok = resp_ok.json()["job_id"]
    final_ok, seen_ok = wait_for_terminal_status(client, job_ok, timeout_s=30, interval_s=0.2)
    assert final_ok is not None
    assert final_ok["status"] == "SUCCESS"
    assert final_ok.get("result") is not None
    assert seen_ok & {"PENDING", "RUNNING", "PROGRESS"}

    ds_fail = "jobs_api_fail"
    pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]}).to_parquet(tmp_path / f"{ds_fail}.parquet")
    resp_fail = client.post(
        "/jobs",
        params={"dataset_id": ds_fail},
        json=AnalysisParams(target_col="missing_target").model_dump(),
    )
    assert resp_fail.status_code == 202
    job_fail = resp_fail.json()["job_id"]
    final_fail, _seen_fail = wait_for_terminal_status(client, job_fail, timeout_s=30, interval_s=0.2)
    assert final_fail is not None
    assert final_fail["status"] == "FAILURE"
    assert final_fail.get("error")
