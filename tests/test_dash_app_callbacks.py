import base64
import io
from types import SimpleNamespace

import dash

from advanced_catdap.frontend import dash_app as dash_mod
from advanced_catdap.service.schema import DatasetMetadata, ColumnInfo


def _sample_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="ds1",
        filename="input.csv",
        file_path="C:/tmp/ds1.parquet",
        n_rows=3,
        n_columns=3,
        columns=[
            ColumnInfo(name="target", dtype="BIGINT", missing_count=0, unique_approx=2),
            ColumnInfo(name="f1", dtype="BIGINT", missing_count=0, unique_approx=3),
            ColumnInfo(name="f2", dtype="VARCHAR", missing_count=0, unique_approx=2),
        ],
    )


def test_handle_file_upload_success(monkeypatch):
    meta = _sample_metadata()
    monkeypatch.setattr(
        dash_mod.client,
        "upload_dataset",
        lambda _file_obj, _filename: meta,
    )
    contents = "data:text/csv;base64," + base64.b64encode(b"target,f1\n1,10\n0,11\n").decode()
    alert, meta_dict, settings = dash_mod.handle_file_upload(contents, "input.csv")
    assert "Loaded: input.csv" in str(alert)
    assert meta_dict["dataset_id"] == "ds1"
    assert settings is not None


def test_handle_file_upload_error(monkeypatch):
    def _raise(_file_obj, _filename):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(dash_mod.client, "upload_dataset", _raise)
    contents = "data:text/csv;base64," + base64.b64encode(b"a,b\n1,2\n").decode()
    alert, meta_dict, settings = dash_mod.handle_file_upload(contents, "broken.csv")
    assert "Error: upload failed" in str(alert)
    assert meta_dict is None
    assert settings is None


def test_submit_job_success(monkeypatch):
    monkeypatch.setattr(dash_mod.client, "submit_job", lambda dataset_id, params: "job-123")
    meta = _sample_metadata().model_dump()
    job_id, params_dict, disabled, msg = dash_mod.submit_job(
        1, meta, "target", "auto", 5, 10, True
    )
    assert job_id == "job-123"
    assert params_dict["target_col"] == "target"
    assert disabled is False
    assert "Analysis started" in str(msg)


def test_submit_job_missing_target():
    meta = _sample_metadata().model_dump()
    result = dash_mod.submit_job(1, meta, "", "auto", 5, 10, True)
    assert "Please select a Target Variable" in str(result[3])


def test_poll_job_completed(monkeypatch):
    monkeypatch.setattr(
        dash_mod.client,
        "get_job_status",
        lambda _job_id: {"status": "SUCCESS", "result": {"feature_importances": []}},
    )
    result, disabled, msg = dash_mod.poll_job(1, "job-123")
    assert result == {"feature_importances": []}
    assert disabled is True
    assert "Completed" in str(msg)


def test_poll_job_failed(monkeypatch):
    monkeypatch.setattr(
        dash_mod.client,
        "get_job_status",
        lambda _job_id: {"status": "FAILURE", "error": "boom"},
    )
    result, disabled, msg = dash_mod.poll_job(1, "job-123")
    assert result is None
    assert disabled is True
    assert "Job Failed" in str(msg)


def test_download_html_report_returns_payload(monkeypatch):
    monkeypatch.setattr(
        dash_mod.client,
        "get_job_status",
        lambda _job_id: {
            "result": {"feature_importances": []},
            "params": {"target_col": "target", "task_type": "auto"},
        },
    )
    monkeypatch.setattr(
        dash_mod.ResultExporter,
        "generate_html_report",
        lambda *_args, **_kwargs: io.BytesIO(b"<html></html>"),
    )
    monkeypatch.delenv("CATDAP_DESKTOP_MODE", raising=False)
    payload, status = dash_mod.download_html_report(
        1,
        {"feature_importances": []},
        "job-123",
        {"filename": "input.csv"},
        None,
        {"target_col": "target", "task_type": "auto"},
        "dark",
    )
    assert payload is not dash.no_update
    assert status is dash.no_update


def test_download_html_report_with_custom_filename(monkeypatch):
    captured = {}
    def _capture_report(result_arg, _meta, **_kwargs):
        captured["result"] = result_arg
        return io.BytesIO(b"<html></html>")

    monkeypatch.setattr(
        dash_mod.ResultExporter,
        "generate_html_report",
        _capture_report,
    )
    monkeypatch.delenv("CATDAP_DESKTOP_MODE", raising=False)
    payload, status = dash_mod.download_html_report(
        1,
        {"feature_importances": []},
        None,
        {"filename": "input.csv"},
        "custom-report",
        {"target_col": "target", "task_type": "auto"},
        "dark",
    )
    assert payload is not dash.no_update
    assert status is dash.no_update
    assert captured["result"]["requested_task_type"] == "auto"


def test_download_html_report_desktop_mode_saved(monkeypatch):
    monkeypatch.setenv("CATDAP_DESKTOP_MODE", "1")
    monkeypatch.setattr(
        dash_mod.client,
        "export_html_report",
        lambda **_kwargs: {"saved": True, "path": "C:/exports/out.html"},
    )
    payload, status = dash_mod.download_html_report(
        1,
        {"feature_importances": []},
        None,
        {"filename": "input.csv"},
        "custom-report",
        {"target_col": "target", "task_type": "auto"},
        "dark",
    )
    assert payload is dash.no_update
    assert "saved" in str(status).lower()


def test_download_html_report_desktop_mode_cancelled(monkeypatch):
    monkeypatch.setenv("CATDAP_DESKTOP_MODE", "1")
    monkeypatch.setattr(
        dash_mod.client,
        "export_html_report",
        lambda **_kwargs: {"saved": False, "reason": "cancelled"},
    )
    payload, status = dash_mod.download_html_report(
        1,
        {"feature_importances": []},
        None,
        {"filename": "input.csv"},
        "custom-report",
        {"target_col": "target", "task_type": "auto"},
        "dark",
    )
    assert payload is dash.no_update
    assert "cancelled" in str(status).lower()


def test_update_deepdive_state_mode(monkeypatch):
    monkeypatch.setattr(dash_mod, "ctx", SimpleNamespace(triggered_id={"type": "deepdive-mode"}))
    state = dash_mod.update_deepdive_state(["select"], [None], [None], {"mode": "top5", "feature": None, "interaction": None})
    assert state["mode"] == "select"


def test_update_deepdive_state_feature(monkeypatch):
    monkeypatch.setattr(dash_mod, "ctx", SimpleNamespace(triggered_id={"type": "deepdive-feat-select"}))
    state = dash_mod.update_deepdive_state(["top5"], ["A"], [None], {"mode": "top5", "feature": None, "interaction": None})
    assert state["feature"] == "A"
