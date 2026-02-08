import sqlite3
from pathlib import Path
from unittest.mock import patch

from advanced_catdap.service.repository import SQLiteJobRepository


def test_save_job_wraps_sqlite_error(tmp_path):
    repo = SQLiteJobRepository(tmp_path / "jobs.db")
    repo.init_storage()
    with patch.object(repo, "_get_connection", side_effect=sqlite3.Error("db down")):
        try:
            repo.save_job("j1", "d1", "PENDING", "{}")
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "Database error" in str(e)


def test_get_job_returns_none_on_sqlite_error(tmp_path):
    repo = SQLiteJobRepository(tmp_path / "jobs.db")
    repo.init_storage()
    with patch.object(repo, "_get_connection", side_effect=sqlite3.Error("db down")):
        assert repo.get_job("j1") is None


def test_update_status_writes_progress_field(tmp_path):
    db = tmp_path / "jobs.db"
    repo = SQLiteJobRepository(db)
    repo.init_storage()
    repo.save_job("j1", "d1", "PENDING", "{}")
    repo.update_status("j1", "RUNNING", progress='{"stage":"x"}')
    row = repo.get_job("j1")
    assert row is not None
    assert row["status"] == "RUNNING"
    assert row["progress"] == '{"stage":"x"}'
