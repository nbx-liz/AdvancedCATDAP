import pytest
from pathlib import Path
import pandas as pd
import duckdb
from unittest.mock import patch
from advanced_catdap.service.dataset_manager import DatasetManager

@pytest.fixture
def manager(tmp_path):
    return DatasetManager(storage_dir=tmp_path)

def test_dataset_lifecycle(manager, tmp_path):
    # 1. Prepare CSV
    csv_file = tmp_path / "data.csv"
    df_in = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_in.to_csv(csv_file, index=False)
    
    # 2. Register
    meta = manager.register_dataset(csv_file)
    assert meta.n_rows == 3
    assert meta.n_columns == 2
    assert meta.columns[0].name == "a"
    
    ds_id = meta.dataset_id
    
    # 3. Preview
    df_prev = manager.get_preview(ds_id, n_rows=2)
    assert len(df_prev) == 2
    assert "a" in df_prev.columns
    
    # 4. Sample
    df_samp = manager.get_sample(ds_id, n_rows=2)
    assert len(df_samp) == 2

    # 5. Row Count
    assert manager._get_row_count(manager.storage_dir / f"{ds_id}.parquet") == 3
    
    # 6. Get Metadata (Re-register logic)
    meta2 = manager.register_dataset(manager.storage_dir / f"{ds_id}.parquet", dataset_id=ds_id)
    assert meta2.n_rows == 3

def test_missing_file_errors(manager):
    with pytest.raises(FileNotFoundError):
        manager.register_dataset("non_existent.csv")
        
    with pytest.raises(FileNotFoundError):
        manager.get_preview("bad_id")
        
    with pytest.raises(FileNotFoundError):
        manager.get_sample("bad_id")

def test_invalid_format(manager, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported format"):
        manager.register_dataset(txt_file)


def test_register_dataset_with_special_column_names(manager, tmp_path):
    csv_file = tmp_path / "special_cols.csv"
    df_in = pd.DataFrame(
        {
            "select": [1, 2, None],
            "a b": ["x", None, "z"],
            "a-b": [10.0, 10.0, 10.0],
            'quote"col': [None, 1, 1],
        }
    )
    df_in.to_csv(csv_file, index=False)

    meta = manager.register_dataset(csv_file)
    by_name = {c.name: c for c in meta.columns}
    assert set(by_name) == {"select", "a b", "a-b", 'quote"col'}
    assert by_name["select"].missing_count == 1
    assert by_name["a b"].missing_count == 1
    assert by_name['quote"col'].missing_count == 1
    assert by_name["a-b"].unique_approx >= 1


def test_quote_identifier_escapes_double_quotes():
    assert DatasetManager._quote_identifier('a"b') == '"a""b"'


def test_register_dataset_cleanup_ignores_remove_error(manager, tmp_path):
    txt_file = tmp_path / "bad.txt"
    txt_file.write_text("hello")
    # Unsupported format triggers exception path and cleanup attempt.
    with patch("advanced_catdap.service.dataset_manager.os.remove", side_effect=OSError("deny")):
        with pytest.raises(ValueError, match="Unsupported format"):
            manager.register_dataset(txt_file, dataset_id="fixed_id")


def test_register_dataset_cleanup_handles_remove_error_after_partial_write(manager, tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_file, index=False)
    with patch.object(manager, "_build_metadata_from_parquet", side_effect=RuntimeError("meta fail")):
        with patch("advanced_catdap.service.dataset_manager.os.remove", side_effect=OSError("deny")):
            with pytest.raises(RuntimeError, match="meta fail"):
                manager.register_dataset(csv_file, dataset_id="fixed_id_2")


def test_get_sample_raises_and_logs_when_duckdb_fails(manager, tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_file, index=False)
    meta = manager.register_dataset(csv_file)

    class _BrokenConn:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("sample failed")
        def close(self):
            return None

    with patch.object(manager, "_get_connection", return_value=_BrokenConn()):
        with pytest.raises(RuntimeError, match="sample failed"):
            manager.get_sample(meta.dataset_id, n_rows=2)


def test_get_dataset_metadata_missing_raises(manager):
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.get_dataset_metadata("missing_id")
