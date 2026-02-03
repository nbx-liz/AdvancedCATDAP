from advanced_catdap.service.dataset_manager import DatasetManager
import pandas as pd
import os
import shutil

def debug():
    manager = DatasetManager(storage_dir="data")
    
    # Check if any .parquet file exists in data/
    files = list(manager.storage_dir.glob("*.parquet"))
    if not files:
        print("No datasets found in data/ to debug.")
        # Create one
        df = pd.DataFrame({"A": [1, 2, 3]})
        df.to_csv("temp_debug.csv", index=False)
        meta = manager.register_dataset("temp_debug.csv")
        print(f"Created debug dataset: {meta.dataset_id}")
        dataset_id = meta.dataset_id
        os.remove("temp_debug.csv")
    else:
        dataset_id = files[0].stem
        print(f"Testing with existing dataset: {dataset_id}")

    print("Attempting get_preview...")
    try:
        preview = manager.get_preview(dataset_id, n_rows=10)
        print("Preview Success:")
        print(preview)
    except Exception as e:
        print(f"Preview Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
