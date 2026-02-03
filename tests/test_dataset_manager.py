import pytest
from pathlib import Path
import pandas as pd
import duckdb
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
