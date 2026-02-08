import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from advanced_catdap.service.job_manager import JobManager
from advanced_catdap.service.schema import AnalysisParams
from integration_helpers import wait_for_terminal_status


class _CapturingExecutor:
    def __init__(self) -> None:
        self.last_cmd = None

    def submit(self, cmd, log_file, env=None):
        self.last_cmd = list(cmd)
        run_env = None
        if env:
            run_env = dict(env)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as out:
            subprocess.Popen(cmd, stdout=out, stderr=out, env=run_env)


@pytest.mark.integration
def test_frozen_worker_command_and_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "jobs.db"
    wrapper = tmp_path / "exe_wrapper.cmd"
    windows_main_path = (repo_root / "scripts" / "windows_main.py").resolve()
    wrapper.write_text(
        f'@echo off\r\n"{sys.executable}" "{windows_main_path}" %*\r\n',
        encoding="utf-8",
    )

    executor = _CapturingExecutor()
    jm = JobManager(db_path=str(db_path), executor=executor)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(wrapper))

    dataset_id = "frozen_worker_ds"
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "x1": [10, 20, 30, 40, 50, 60],
        }
    )
    df.to_parquet(tmp_path / f"{dataset_id}.parquet")

    job_id = jm.submit_job(dataset_id, AnalysisParams(target_col="target"))
    final_status, _seen = wait_for_terminal_status(jm, job_id, timeout_s=30, interval_s=0.2)

    assert executor.last_cmd is not None
    cmd = executor.last_cmd
    assert "--worker" in cmd
    assert "--job-id" in cmd
    assert "--dataset-id" in cmd
    assert "--params" in cmd
    assert "--db-path" in cmd
    assert "--data-dir" in cmd

    assert final_status is not None
    assert final_status["status"] in {"SUCCESS", "FAILURE"}
    assert final_status["status"] != "PENDING"

    log_file = tmp_path / "jobs_logs" / f"{job_id}.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8", errors="ignore")
    assert "Starting AdvancedCATDAP (Dash version)..." not in log_text
